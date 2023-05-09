import argparse
import os
from datetime import datetime
import time
import os
import pdb
import sys
import numpy as np
from pathlib import Path
from queue import Queue

import av
import cv2
import face_recognition
import speech_recognition as sr
import tellopy
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

# Load model
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
input_face_pairs = [('colin.jpg', 'colin')]
# input_face_pairs = []
known_face_encodings = []
known_face_names = []

for path, name in input_face_pairs:
    if not os.path.exists(path):
        print('Could not find image at path:', path)
        continue
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

# grab the raw bytes and push it into the thread safe queue.
def record_callback(_, audio:sr.AudioData) -> None:
    data = audio.get_raw_data()
    data_queue.put(data)

recorder.listen_in_background(audio_source, record_callback, phrase_time_limit=2)

# video source = 'drone' or 'webcam'
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

follow_data = {
    'known_width':known_width,
    'focal_length':focal_length,
    'rows':rows,
    'cols':cols,
    'frame_width':frame_width,
    'adjacency_matrix':adjacency_matrix,
    'start_node':start_node}

if video_source == 'drone':
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
        while True:
            for raw_frame in container.decode(video=0):
                if 0 < frame_skip:
                    frame_skip = frame_skip - 1
                    continue
                start_time = time.time()
                frame = np.array(raw_frame.to_image())
                frame_width = frame.shape[1]
                now = datetime.utcnow()
                if not data_queue.empty():
                    output_text, transcription, phrase_time, last_sample = utils.process_audio(data_queue, audio_source, phrase_time, last_sample, transcription, audio_model, temp_file)
                
                rotation_angle, euclidean_distance = tracking.track(frame, model, tracker_list, checked_people, known_face_encodings, known_face_names, follow_data, args.follow) 
                
                if rotation_angle:
                    if rotation_angle < 0:
                        print('counterclockwise 10')
                    if rotation_angle > 0:
                        print('clockwise 10')

                if output_text:
                    try:
                        command.queue_text(output_text, queue)
                    except:
                        _ = None
                    # print(command.queue_text(output_text, queue))
                    output_text = None

                while not queue.empty():
                    j = queue.get(0)
                    k = 0
                    if j == "forward" or j == "back" or j == "clockwise" or j == "counterclockwise":
                        k = queue.get(0)
                    print('zoop', j, k)
                    # command.commands(j, k, drone)
                
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

else:
    queue = Queue()
    while True:
        output_text = None
        # Pull raw recorded audio from the queue.
        if not data_queue.empty():
            output_text, transcription, phrase_time, last_sample = utils.process_audio(data_queue, audio_source, phrase_time, last_sample, transcription, audio_model, temp_file)

        # process frame
        ret, frame = video_capture.read()
        rotation_angle, distance = tracking.track(frame, model, tracker_list, checked_people, known_face_encodings, known_face_names, follow_data, args.follow)
        if rotation_angle or rotation_angle == 0:
            if rotation_angle < 0:
                print('counterclockwise 10')
                time.sleep(2)
            if rotation_angle > 0:
                print('clockwise 10')
                time.sleep(2)
            if distance > 2:
                print('forward 20')
                time.sleep(2)
        
        if output_text:
            command.queue_text(output_text, queue)

        while not queue.empty():
            j = queue.get(0)
            k = 0
            if j == "forward" or j == "back" or j == "clockwise" or j == "counterclockwise" and not queue.empty():
                k = queue.get(0)
            command.commands(j, k, None)