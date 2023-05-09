import numpy as np
import speech_recognition as sr
import whisper

from queue import Queue
from tempfile import NamedTemporaryFile
from sys import platform
from datetime import datetime, timedelta
import torch
import os
import io

def depth_estimation(known_width, focal_length, pixel_width):
    return (known_width * focal_length) / pixel_width

def process_audio(data_queue, audio_source, phrase_time, last_sample, transcription, audio_model, temp_file):
    now = datetime.utcnow()
    phrase_complete = False
    # If enough time has passed between recordings, consider the phrase complete.
    # Clear the current working audio buffer to start over with the new data.
    if phrase_time and now - phrase_time > timedelta(seconds=3):
        last_sample = bytes()
        phrase_complete = True
    # This is the last time we received new audio data from the queue.
    phrase_time = now

    # Concatenate our current audio data with the latest audio data.
    while not data_queue.empty():
        data = data_queue.get()
        last_sample += data

    # Use AudioData to convert the raw data to wav data.
    audio_data = sr.AudioData(last_sample, audio_source.SAMPLE_RATE, audio_source.SAMPLE_WIDTH)
    wav_data = io.BytesIO(audio_data.get_wav_data())

    # Write wav data to the temporary file as bytes.
    with open(temp_file, 'w+b') as f:
        f.write(wav_data.read())

    # Read the transcription.
    result = audio_model.transcribe(temp_file, fp16=torch.cuda.is_available())
    text = result['text'].strip()

    # If we detected a pause between recordings, add a new item to our transcripion.
    # Otherwise edit the existing one.
    if phrase_complete:
        transcription.append(text)
    else:
        transcription[-1] = text

    # Clear the console to reprint the updated transcription.
    os.system('cls' if os.name=='nt' else 'clear')
    output_text = ' '.join(transcription)
    for line in transcription:
        print(line)
    # Flush stdout.
    print('', end='', flush=True) 
    if phrase_complete:
        transcription = ['']
    return output_text, transcription, phrase_time, last_sample

def set_up_recorder():
    model = 'base'

    # The last time a recording was retreived from the queue.
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feauture where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = 1000
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramtically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False

    # Prevents permanent application hang and crash by using the wrong Microphone
    if 'linux' in platform:
        mic_name = 'pulse'
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")   
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    audio_source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        audio_source = sr.Microphone(sample_rate=16000)
        
    # Load / Download model
    model = model
    if model != "large":
        model = model + ".en"
    #import pdb
    #pdb.set_trace()
    audio_model = whisper.load_model(model)

    temp_file = NamedTemporaryFile().name

    with audio_source:
        recorder.adjust_for_ambient_noise(audio_source)
    return recorder, data_queue, audio_source, audio_model, temp_file

def generate_adjacency_matrix(rows, cols):
    adjacency_matrix = np.zeros((rows * cols, rows * cols))
    for i in range(rows):
        for j in range(cols):
            node = i * cols + j
            # check above
            if i < rows - 1:
                adjacency_matrix[node][node + cols] = 1
            # check right
            if j < cols - 1:
                adjacency_matrix[node][node + 1] = 1
                # check up-right
                if i < rows - 1:
                    adjacency_matrix[node][node + cols + 1] = np.sqrt(2)
            # check left
            if j > 0:
                adjacency_matrix[node][node - 1] = 1
                # check up-left
                if i < rows - 1:
                    adjacency_matrix[node][node + cols - 1] = np.sqrt(2)
    # this is the start node. it is connected to every node in the first row. it is positioned in the middle of the grid row, and is euclidean distance away from each node
    # start_node_x = (cols-1) / 2
    # start_node_y = 0
    for i in range(2):
        for j in range(cols):
            node = i * cols + j
            # adjacency_matrix[0][node] = 1
            node_x_diff = j - (cols-1) / 2
            node_y_diff = i
            distance = np.sqrt(node_x_diff ** 2 + node_y_diff ** 2)
            adjacency_matrix[0][node] = distance
    return adjacency_matrix

def find_closest_node(box, distance, rows, cols, frame_width, focal_length):
    start_node_x = (cols-1) / 2
    # find distance from center of frame: box center - frame center
    pixel_distance = (box[0] + box[2]) / 2 - (frame_width / 2)
    # meters from center of frame
    center_dist = distance * pixel_distance / focal_length
    # find closest column
    col = round(center_dist + start_node_x)
    # find closest row
    row = round(distance)
    # convert row and column to node
    node = row * cols + col

    # node cant be less than 0 or greater than the number of nodes
    if node < 0:
        node = 0
    elif node > rows * cols - 1:
        node = rows * cols - 1
    return node


def dijkstras(matrix, start_node, target_node, cols):
    # initialize distance array
    distance = np.zeros(matrix.shape[0])
    for i in range(distance.shape[0]):
        distance[i] = float('inf')
    distance[start_node] = 0

    # initialize visited array
    visited = np.zeros(matrix.shape[0])

    # initialize previous array
    previous = np.zeros(matrix.shape[0])

    # initialize queue
    queue = []
    queue.append(start_node)

    while queue:
        # find node with minimum distance
        min_distance = float('inf')
        min_index = 0
        for i in range(len(queue)):
            if distance[queue[i]] < min_distance:
                min_distance = distance[queue[i]]
                min_index = i
        node = queue.pop(min_index)
        visited[node] = 1

        # check all neighbors of node
        for i in range(matrix.shape[0]):
            if matrix[node][i] != 0 and visited[i] == 0:
                # update distance
                if distance[i] > distance[node] + matrix[node][i]:
                    distance[i] = distance[node] + matrix[node][i]
                    previous[i] = node
                if i not in queue:  # Check if the neighbor is already in the queue before appending it
                    queue.append(i)

    # backtrack to find shortest path
    path = []
    node = target_node
    while node != 0:
        path.append(node)
        node = int(previous[node])
    # path.append(0)
    path.reverse()

    straight_shot_node = start_node
    all_cols = []
    for i in range(len(path)):
        col = path[i] % cols
        all_cols.append(col)
        if all_cols != sorted(all_cols) and all_cols != sorted(all_cols, reverse=True):
            break
        straight_shot_node = path[i]

    return path, straight_shot_node


def determine_commands(straight_shot_node, focal_length, cols, start_node):
    if straight_shot_node == start_node:
        return 0, 0
    # determine the amount of pixels we need to rotate to face the straight shot node
    straight_col = straight_shot_node % cols
    straight_row = int(straight_shot_node / cols)
    center_col = int(cols / 2)
    col_diff = straight_col - center_col
    row_diff = straight_row
    euclidean_distance = np.sqrt(col_diff ** 2 + row_diff ** 2)
    pixel_rotation = int(col_diff / euclidean_distance * focal_length)

    # calculate rotation angle x = 90 - arctan(col_diff / row_diff)
    if col_diff == 0:
        rotation_angle = 0
    else:
        rotation_angle = 90 - np.arctan(abs(row_diff / col_diff)) * 180 / np.pi
        if col_diff < 0:
            rotation_angle *= -1
    # roration angle is reversed because camera is mirorred. positive == clockwise, negative == counterclockwise
    rotation_angle *= -1

    return rotation_angle, euclidean_distance
