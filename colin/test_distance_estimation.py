# adapted from https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam.py

import face_recognition
import cv2
import numpy as np

video_capture = cv2.VideoCapture(0)

# make a 7 by 10 grid. each dot is 1 meter away from the surrounding nodes. make an adjacency matrix of the grid
rows = 4
cols = 3

start_node = int((cols-1) / 2)

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
                adjacency_matrix[node][node + cols + 1] = 1
        # check left
        if j > 0:
            adjacency_matrix[node][node - 1] = 1
            # check up-left
            if i < rows - 1:
                adjacency_matrix[node][node + cols - 1] = 1

# # prepend and append a row and column of zeros to the adjacency matrix
# adjacency_matrix = np.insert(adjacency_matrix, 0, 0, axis=0)
# adjacency_matrix = np.insert(adjacency_matrix, 0, 0, axis=1) 

# this is the start node. it is connected to every node in the first row. it is positioned in the middle of the grid row, and is euclidean distance away from each node
start_node_x = (cols-1) / 2
start_node_y = 0
for i in range(2):
    for j in range(cols):
        node = i * cols + j
        # adjacency_matrix[0][node] = 1
        node_x_diff = j - (cols-1) / 2
        node_y_diff = i
        distance = np.sqrt(node_x_diff ** 2 + node_y_diff ** 2)
        adjacency_matrix[0][node] = distance

def dijkstras(matrix, start_node, target_node):
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

    # check path to set [up, down, left, right] directions taken
    directions = [False, False, False, False]
    straight_shot_node = start_node
    all_cols = []
    for i in range(len(path)):
        # row = int(path[i] / cols)
        col = path[i] % cols
        all_cols.append(col)
        if all_cols != sorted(all_cols) and all_cols != sorted(all_cols, reverse=True):
            break
        straight_shot_node = path[i]

    return path, straight_shot_node


    # this function takes a pixel distance from the center of the frame and a distance in meters from the camera and returns the node that is closest to the the object
def find_closest_node(box, distance, dim):
    # find distance from center of frame: box center - frame center
    pixel_distance = (box[1] + box[3]) / 2 - (frame_width / 2)
    # meters from center of frame
    center_dist = distance * pixel_distance / focal_length
    # find closest column
    col = round(center_dist + start_node_x)
    # find closest row
    row = round(distance)
    # convert row and column to node
    node = row * cols + col
    # print(node, pixel_distance, center_dist, distance, dim)
    return node

def determine_commands(straight_shot_node):
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

    return pixel_rotation, euclidean_distance


# if you want to detect yourself, add (img_path, name) tuples to the input_face_pairs list below
input_face_pairs = []
known_face_encodings = []
known_face_names = []

for path, name in input_face_pairs:
    image = face_recognition.load_image_file(path)
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)
    known_face_names.append(name)

face_locations = []
face_encodings = []
face_names = []

vid = cv2.VideoCapture(0)


def calibrate(known_width, known_distance, pixel_width):
    focal_length = (pixel_width * known_distance) / known_width
    return focal_length

def depth_estimation(known_width, focal_length, pixel_width):
    return (known_width * focal_length) / pixel_width

# calibrate camera
known_width = 0.2 # meters
known_distance = 1 # meters
pixel_width = 250 # pixels
focal_length = calibrate(known_width, known_distance, pixel_width)


process = 0
while(True):

    ret, frame = video_capture.read()
    if process % 2 == 0:
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if face_distances:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
            face_names.append(name)

    # estimate depth of face using known face size
    def distance_to_camera(knownWidth, focalLength, perWidth):
        return (knownWidth * focalLength) / perWidth
    
    # print(depth)

    # print width of entire frame
    frame_width = frame.shape[1]
    num_dots = 5
    dot_dist = int(frame_width / (num_dots + 2))



    distance = depth_estimation(known_width, focal_length, face_locations[0][1] - face_locations[0][3])
    target_node = find_closest_node(face_locations[0], distance, rows)
    path, straight_shot_node = dijkstras(adjacency_matrix, start_node, target_node)


    pixel_rotation, euclidean_distance = determine_commands(straight_shot_node)
    

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    process += 1

vid.release()
cv2.destroyAllWindows()
