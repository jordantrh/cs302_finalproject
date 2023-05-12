'''
This file contains various helper functions for the main script. Specifically, it contains functions for:
- depth estimation
- audio processing, adapted from: https://github.com/davabase/whisper_real_time
- shortest path algorithm for following
'''


from datetime import datetime, timedelta
import io
import numpy as np
import os
from queue import Queue
import speech_recognition as sr
from sys import platform
from tempfile import NamedTemporaryFile
import torch
import whisper


# estimate depth of object from known width, focal length, and pixel width
def depth_estimation(known_width, focal_length, pixel_width):
    return (known_width * focal_length) / pixel_width


# process audio data from the queue to text
def process_audio(data_queue, audio_source, phrase_time, last_sample, transcription, audio_model, temp_file):
    now = datetime.utcnow()
    phrase_complete = False
    # If enough time has passed between recordings, consider the phrase complete.
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
    if phrase_complete:
        transcription.append(text)
    else:
        transcription[-1] = text

    # Clear the console to reprint the updated transcription.
    os.system('cls' if os.name=='nt' else 'clear')
    output_text = ' '.join(transcription)
    for line in transcription:
        print('STT: '+line)
    
    if phrase_complete:
        transcription = ['']
    return output_text, transcription, phrase_time, last_sample

# set up the audio recorder
def set_up_recorder():
    model = 'base'

    data_queue = Queue()
    
    recorder = sr.Recognizer()
    recorder.energy_threshold = 1000
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
    
    audio_model = whisper.load_model(model)

    temp_file = NamedTemporaryFile().name

    with audio_source:
        recorder.adjust_for_ambient_noise(audio_source)
    return recorder, data_queue, audio_source, audio_model, temp_file

# generate adjacency matrix for shortest path algorithm using set number of rows and columns. nodes are 1 meter apart in both x and y dims
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
    
    # we start at the center node of the bottom row, so we need to set the distance to the center node to all surrounding nodes
    for i in range(2):
        for j in range(cols):
            node = i * cols + j
            node_x_diff = j - (cols-1) / 2
            node_y_diff = i
            distance = np.sqrt(node_x_diff ** 2 + node_y_diff ** 2)
            adjacency_matrix[0][node] = distance

    return adjacency_matrix


# find the node closest to the center of an object bounding box
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


# find the shortest path from start_node to target_node using Dijkstra's algorithm
def dijkstras(matrix, start_node, target_node, cols):
    # initialize distance, visited, and previous arrays
    distance = np.zeros(matrix.shape[0])
    for i in range(distance.shape[0]):
        distance[i] = float('inf')
    distance[start_node] = 0

    visited = np.zeros(matrix.shape[0])

    previous = np.zeros(matrix.shape[0])

    queue = []
    queue.append(start_node)

    # while queue is not empty
    while queue:
        min_distance = float('inf')
        min_index = 0
        # find node with minimum distance
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
                if i not in queue:
                    queue.append(i)

    # backtrack to find shortest path
    path = []
    node = target_node
    while node != 0:
        path.append(node)
        node = int(previous[node])
    path.reverse()

    # find the node that is the furthest from the start node that is still in a straight line
    # we will travel straight to this node
    straight_shot_node = start_node
    all_cols = []
    for i in range(len(path)):
        col = path[i] % cols
        all_cols.append(col)
        if all_cols != sorted(all_cols) and all_cols != sorted(all_cols, reverse=True):
            break
        straight_shot_node = path[i]

    return path, straight_shot_node

# determine the commands to send to the drone to follow the target
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
