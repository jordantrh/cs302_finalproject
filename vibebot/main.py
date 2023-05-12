'''
CS302 Final Project: VibeBot
Colin Smith, Austin Tran, Jordan Huff, Matthew Webb
3/11/23

This is the main file for a program that tracks objects, detects people, tracks a specific person, and calculates a shortest path to follow them. 
Also, it records audio, transcribes it, and executes commands based on the transcription. This program works with a webcam or a drone.
'''

import argparse
import av
import cv2
from datetime import datetime
import face_recognition
import numpy as np
from queue import Queue
import os
from pathlib import Path
import speech_recognition as sr
import sys
import tellopy
import time
import torch

from ultralytics.nn.autobackend import AutoBackend
from trackers.multi_tracker_zoo import create_tracker

import download_weights
import utils
import tracking
import command

# argument parsing: drone or webcam, follow
parser = argparse.ArgumentParser()
parser.add_argument('--source', '-s', type=str, default='webcam', help='video source: drone or webcam')
parser.add_argument('--follow', '-f', type=str, default='', help='follow: put in a name of someone to follow')
args = parser.parse_args()

# download weights if they don't exist
download_weights.download_weights()

# set up path arguments for yolov5 and strongsort
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'
if str(ROOT / 'trackers' / 'strongsort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strongsort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
yolo_weights = WEIGHTS / 'yolov8s-seg.pt'
reid_weights = WEIGHTS / 'osnet_x0_25_msmt17.pt'
tracking_method = 'strongsort'
tracking_config = ROOT / 'trackers' / tracking_method / 'configs' / (tracking_method + '.yaml')
name = 'exp'
half = False

# Load yolo model
device = torch.device('cpu')
is_seg = '-seg' in str(yolo_weights)
model = AutoBackend(yolo_weights, device=device, dnn=False, fp16=half)


# create tracker instance
tracker_list = []
tracker = create_tracker(tracking_method, tracking_config, reid_weights, device, half)
tracker_list.append(tracker, )
if hasattr(tracker_list[0], 'model'):
    if hasattr(tracker_list[0].model, 'warmup'):
        tracker_list[0].model.warmup()

# calibrate camera
def calibrate(known_width, known_distance, pixel_width):
    focal_length = (pixel_width * known_distance) / known_width
    return focal_length

known_width = 0.4 # meters
known_distance = 1 # meters
pixel_width = 800 # pixels
focal_length = calibrate(known_width, known_distance, pixel_width)


# if you want to detect yourself, add (img_path, name) tuples to the input_face_pairs list below
input_face_pairs = []
known_face_encodings = []
known_face_names = []

# if pics dir doesn't exist, create it
if not os.path.exists('./pics'):
    os.mkdir('./pics')

# iterate over all pics in ./pics dir
for filename in os.listdir('./pics'):
    print(filename)
    if filename.endswith('.jpeg'):
        # split on period to get name
        name = filename.split('.')[0]
        input_face_pairs.append((os.path.join('pics', filename), name))
print(input_face_pairs)

# load face images and generate encodings for pictures in ./pics dir
for path, name in input_face_pairs:
    if not os.path.exists(path):
        print('Could not find image at path:', path)
        continue
    print('Loading image at path:', path, 'with name:', name)
    image = face_recognition.load_image_file(path)
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)
    known_face_names.append(name)

face_locations = []
face_encodings = []
face_names = []

# keeps track of person index to name mappings
checked_people = {}

# set up recording and transcription
recorder, data_queue, audio_source, audio_model, temp_file = utils.set_up_recorder()
transcription = ['']
phrase_time = None
last_sample = bytes()

def record_callback(_, audio:sr.AudioData) -> None:
    data = audio.get_raw_data()
    data_queue.put(data)

recorder.listen_in_background(audio_source, record_callback, phrase_time_limit=2)

# set up video source, which can be either 'drone' or 'webcam'
video_source = args.source
if video_source == 'webcam':
    video_capture = cv2.VideoCapture(0)
    ret, frame = video_capture.read()
    frame_width = frame.shape[1]
elif video_source == 'drone':
    drone = tellopy.Tello()
    frame_width = 960

# the following is needed for tracking
rows = 7
cols = 3 # must be odd
start_node = int((cols-1) / 2)
adjacency_matrix = utils.generate_adjacency_matrix(rows, cols)

# follow_data is a dictionary that contains all the data needed for the shortest path algorithm
follow_data = {
    'known_width':known_width,
    'focal_length':focal_length,
    'rows':rows,
    'cols':cols,
    'frame_width':frame_width,
    'adjacency_matrix':adjacency_matrix,
    'start_node':start_node}

# main loop over video frames. split into drone and webcam cases
prev_frame = None
if video_source == 'drone':
    # connect to drone
    drone.connect()
    drone.wait_for_connection(60.0)
    queue = Queue()
    frame_skip = 10
    output_text = None
    try:
        retry = 3
        container = None
        while container is None and 0 < retry:
            retry -= 1
            try:
                container = av.open(drone.get_video_stream())
            except av.AVError as ave:
                print(ave)
                print('retry...')

        # main loop over video frames
        while True:
            for raw_frame in container.decode(video=0):
                if 0 < frame_skip:
                    frame_skip = frame_skip - 1
                    continue
                # collect start time so that we can skip frames if we are behind
                start_time = time.time()
                frame = np.array(raw_frame.to_image())
                frame_width = frame.shape[1]
                now = datetime.utcnow()

                # if there is audio data in the queue, process it
                if not data_queue.empty():
                    output_text, transcription, phrase_time, last_sample = utils.process_audio(data_queue, audio_source, phrase_time, last_sample, transcription, audio_model, temp_file)
                rotation_angle = None 

                # process frame: yolo object detection, strongsort tracking, shortest path calculation, and drone commands for following specific person (if applicable)
                prev_frame, rotation_angle, euclidean_distance = tracking.track(frame, prev_frame, model, tracker_list, checked_people, known_face_encodings, known_face_names, follow_data, args.follow) 

                # print out rotation angle and distance to person if we are following someone
                if rotation_angle or rotation_angle == 0:
                    print(f'Track {args.follow}: rotate {round(rotation_angle)} degrees and travel {round(distance, 3)} meters forward')
                
                # if we have audio output text, queue it up for drone commands
                if output_text:
                    try:
                        command.queue_text(output_text, queue)
                    except:
                        _ = None
                    
                    output_text = None

                # execute drone commands
                while not queue.empty():
                    j = queue.get(0)
                    k = 0
                    if j == "forward" or j == "back" or j == "clockwise" or j == "counterclockwise" or j == "left" or j == "right" or j == "up" or j == "down" and not queue.empty():
                        k = queue.get(0)
                    command.commands(j, k, drone)

                # skip frames if we are behind 
                if raw_frame.time_base < 1.0/60:
                        time_base = 1.0/60
                else:
                    time_base = raw_frame.time_base
                frame_skip = int((time.time() - start_time)/time_base)

    except KeyboardInterrupt:
        print('Interrupted')
        try:
            drone.land()
        except SystemExit:
            os._exit(130)

# webcam case
else:
    queue = Queue()
    # main loop over video frames
    while True:
        output_text = None
        
        # if there is audio data in the queue, process it
        if not data_queue.empty():
            output_text, transcription, phrase_time, last_sample = utils.process_audio(data_queue, audio_source, phrase_time, last_sample, transcription, audio_model, temp_file)

        # process frame: yolo object detection, strongsort tracking, shortest path calculation, and drone commands for following specific person (if applicable)
        ret, frame = video_capture.read()
        prev_frame, rotation_angle, distance = tracking.track(frame, prev_frame, model, tracker_list, checked_people, known_face_encodings, known_face_names, follow_data, args.follow)
        
        # print out rotation angle and distance to person if we are following someone
        if rotation_angle or rotation_angle == 0:
            print(f'Track {args.follow}: rotate {round(rotation_angle)} degrees and travel {round(distance, 3)} meters forward')
        
        # if we have audio output text, queue it up for drone commands
        if output_text:
            command.queue_text(output_text, queue)

        # execute drone commands
        while not queue.empty():
            j = queue.get(0)
            k = 0
            if j == "forward" or j == "back" or j == "clockwise" or j == "counterclockwise" or j == "left" or j == "right" or j == "up" or j == "down" and not queue.empty():
                k = queue.get(0)
            command.commands(j, k, None)