import cv2
import os
import pdb
import sys
import platform
import numpy as np
from pathlib import Path
import torch

import face_recognition

import download_weights

# download weights if they don't exist
download_weights.download_weights()

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT / 'trackers' / 'strongsort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strongsort'))  # add strong_sort ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.utils.checks import check_imgsz, check_imshow
from ultralytics.yolo.utils.ops import Profile, non_max_suppression, scale_boxes, process_mask, process_mask_native
from ultralytics.yolo.utils.plotting import Annotator, colors
from ultralytics.yolo.data.augment import LetterBox

from trackers.multi_tracker_zoo import create_tracker

source = '0' 
yolo_weights = WEIGHTS / 'yolov8s-seg.pt'
reid_weights = WEIGHTS / 'osnet_x0_25_msmt17.pt'
tracking_method = 'strongsort'
tracking_config = ROOT / 'trackers' / tracking_method / 'configs' / (tracking_method + '.yaml')
imgsz = (640, 640)
conf_thres = 0.25
iou_thres = 0.45
max_det = 1000
classes = None
agnostic_nms = False
augment = False
project = ROOT / 'runs' / 'track'
name = 'exp'
exist_ok = False
line_thickness = 2
hide_labels = False
hide_conf = False
hide_class = False
half = False
vid_stride = 1
retina_masks = False




source = str(source)
is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)

# Directories
exp_name = yolo_weights.stem
exp_name = name if name else exp_name + "_" + reid_weights.stem

# Load model
device = torch.device('cpu')
is_seg = '-seg' in str(yolo_weights)
model = AutoBackend(yolo_weights, device=device, dnn=False, fp16=half)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_imgsz(imgsz, stride=stride)  # check image size

# Dataloader
bs = 1
show_vid = check_imshow(warn=True)

vid_path, vid_writer, txt_path = [None] * bs, [None] * bs, [None] * bs
model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup

# Create as many strong sort instances as there are video sources
tracker_list = []
for i in range(bs):
    tracker = create_tracker(tracking_method, tracking_config, reid_weights, device, half)
    tracker_list.append(tracker, )
    if hasattr(tracker_list[i], 'model'):
        if hasattr(tracker_list[i].model, 'warmup'):
            tracker_list[i].model.warmup()
outputs = [None] * bs

seen, windows, dt = 0, [], (Profile(), Profile(), Profile(), Profile())
curr_frames, prev_frames = [None] * bs, [None] * bs


def calibrate(known_width, known_distance, pixel_width):
    focal_length = (pixel_width * known_distance) / known_width
    return focal_length

def depth_estimation(known_width, focal_length, pixel_width):
    return (known_width * focal_length) / pixel_width

# calibrate camera
known_width = 0.4 # meters
known_distance = 1 # meters
pixel_width = 800 # pixels
focal_length = calibrate(known_width, known_distance, pixel_width)


# if you want to detect yourself, add (img_path, name) tuples to the input_face_pairs list below
#input_face_pairs = [('colin.jpg', 'colin')]
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

checked_people = {}

def transform(im0):
    im = np.stack([LetterBox(imgsz, pt, stride=stride)(image=x) for x in im0])
    im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
    im = np.ascontiguousarray(im)  # contiguous
    return im 

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
                adjacency_matrix[node][node + cols + 1] = np.sqrt(2)
        # check left
        if j > 0:
            adjacency_matrix[node][node - 1] = 1
            # check up-left
            if i < rows - 1:
                adjacency_matrix[node][node + cols - 1] = np.sqrt(2)


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

video_capture = cv2.VideoCapture(0)
ret, frame = video_capture.read()
frame_width = frame.shape[1]

process = 0
while True:
    ret, frame = video_capture.read()
    im = transform([frame])
    path = source
    im0s = [frame]
    vid_cap = None
    s = ''

    # visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
    with dt[0]:
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

    # Inference
    with dt[1]:
        preds = model(im, augment=augment)

    # Apply NMS
    with dt[2]:
        if is_seg:
            masks = []
            p = non_max_suppression(preds[0], conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nc = (preds[0].shape[1] - 32 - 4))
            proto = preds[1][-1]
        else:
            p = non_max_suppression(preds, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        
    # Process detections
    for i, det in enumerate(p):  # detections per image
        seen += 1
        p, im0 = path[i], im0s[i].copy()
        p = Path(p)  # to Path
        s += f'{i}: '
        txt_file_name = p.name
        curr_frames[i] = im0

        s += '%gx%g ' % im.shape[2:]  # print string

        annotator = Annotator(im0, line_width=line_thickness, example=str(names))
        
        if hasattr(tracker_list[i], 'tracker') and hasattr(tracker_list[i].tracker, 'camera_update'):
            if prev_frames[i] is not None and curr_frames[i] is not None:  # camera motion compensation
                tracker_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

        if det is not None and len(det):
            if is_seg:
                shape = im0.shape
                # scale bbox first the crop masks
                if retina_masks:
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], shape).round()  # rescale boxes to im0 size
                    masks.append(process_mask_native(proto[i], det[:, 6:], det[:, :4], im0.shape[:2]))  # HWC
                else:
                    masks.append(process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True))  # HWC
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], shape).round()  # rescale boxes to im0 size
            else:
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

            # Print results
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # pass detections to strongsort
            with torch.no_grad():
                with dt[3]:
                    outputs[i] = tracker_list[i].update(det.cpu(), im0)

            # draw boxes for visualization
            if len(outputs[i]) > 0:
                
                if is_seg:
                    # Mask plotting
                    annotator.masks(
                        masks[i],
                        colors=[colors(x, True) for x in det[:, 5]],
                        im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(device).permute(2, 0, 1).flip(0).contiguous() /
                        255 if retina_masks else im[i]
                    )
                
                for j, (output) in enumerate(outputs[i]):
                    
                    bbox = output[0:4]
                    id = output[4]
                    cls = output[5]
                    conf = output[6]

                    # face detection
                    face_check = False

                    # only check once per person
                    if names[int(output[5])] == 'person' and int(output[4]) not in checked_people:
                        # if we havent checked for faces yet
                        if not face_check:
                            face_locations = face_recognition.face_locations(frame)
                            face_check = True
                            # iterate over detected faces, if the location overlaps with the person, encode and compare
                            for (top, right, bottom, left) in face_locations:
                                if (output[0] < left < output[2]) and (output[1] < top < output[3]):
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
                                    checked_people[int(output[4])] = name
                                    break

                    # add distance estimation
                    if names[int(cls)] == 'person':
                        pixels = bbox[2] - bbox[0]
                        distance = depth_estimation(known_width, focal_length, pixels)
                        # distance = depth_estimation(known_width, focal_length, face_locations[0][1] - face_locations[0][3])
                        tracking = False
                        if tracking:
                            target_node = find_closest_node(bbox, distance, rows)
                            path, straight_shot_node = dijkstras(adjacency_matrix, start_node, target_node)
                            pixel_rotation, euclidean_distance = determine_commands(straight_shot_node)
                            print(pixels, distance, pixel_rotation, euclidean_distance, straight_shot_node, path)

                    if show_vid:
                        c = int(cls)  # integer class
                        id = int(id)  # integer id
                        # if c is a person, check if we have a name for them
                        if names[c] == 'person' and id in checked_people:
                            name = checked_people[id]
                        else:
                            name = names[c]
                        label = None if hide_labels else (f'{id} {name}' if hide_conf else \
                            (f'{id} {conf:.2f}' if hide_class else f'{id} {name} {conf:.2f}'))
                        color = colors(c, True)
                        annotator.box_label(bbox, label, color=color)
        else:
            pass
            
        # Stream results
        im0 = annotator.result()
        if show_vid:
            if platform.system() == 'Linux' and p not in windows:
                windows.append(p)
                cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
            cv2.imshow(str(p), im0)
            if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                exit()

        prev_frames[i] = curr_frames[i]
        
    # Print total time (preprocessing + inference + NMS + tracking)
    #LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{sum([dt.dt for dt in dt if hasattr(dt, 'dt')]) * 1E3:.1f}ms")