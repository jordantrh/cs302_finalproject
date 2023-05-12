'''
Functions for tracking people using yolov8. Adapted from: https://github.com/mikel-brostrom/yolov8_tracking
'''

import cv2
import face_recognition
import numpy as np
import torch
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.utils.ops import Profile, non_max_suppression, scale_boxes, process_mask
from ultralytics.yolo.utils.plotting import Annotator, colors
from ultralytics.yolo.data.augment import LetterBox
import utils


# transform image to be compatible with yolov8
def transform(im0, imgsz, pt, stride):
    im = np.stack([LetterBox(imgsz, pt, stride=stride)(image=x) for x in im0])
    im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
    im = np.ascontiguousarray(im)  # contiguous
    return im 

# track people using yolov8
def track(frame, prev_frame, model, tracker_list, checked_people, known_face_encodings, known_face_names, follow_data, target_person):

    # set up way too many variables
    conf_thres = 0.25
    iou_thres = 0.45
    max_det = 1000
    classes = None
    agnostic_nms = False
    imgsz = (640, 640)
    outputs = [None]
    curr_frames = [None]
    prev_frames = [prev_frame]
    stride, names, pt = model.stride, model.names, model.pt
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile(), Profile())
    im = transform([frame], imgsz, pt, stride)
    im0s = [frame]
    s = ''
    rotation_angle = None
    euclidean_distance = None

    # send image to device (or in this case cpu)
    with dt[0]:
        im = torch.from_numpy(im).to(torch.device('cpu'))
        im = im.half() if False else im.float() 
        im /= 255.0
        if len(im.shape) == 3:
            im = im[None] 

    # run image through model
    with dt[1]:
        preds = model(im, augment=False)

    # Apply NMS
    with dt[2]:
        masks = []
        p = non_max_suppression(preds[0], conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nc = (preds[0].shape[1] - 32 - 4))
        proto = preds[1][-1]
        
    # Process detections
    for i, det in enumerate(p):  # detections per image
        seen += 1
        im0 = im0s[i].copy()
        s += f'{i}: '
        curr_frames[i] = im0

        s += '%gx%g ' % im.shape[2:] 

        # this is for visualization of bounding boxes
        annotator = Annotator(im0, line_width=2, example=str(names))
        
        # have tracker update camera position
        if hasattr(tracker_list[i], 'tracker') and hasattr(tracker_list[i].tracker, 'camera_update'):
            if prev_frames[i] is not None and curr_frames[i] is not None:
                tracker_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

        # mask detection and scale boxes
        if det is not None and len(det):
            shape = im0.shape
            masks.append(process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)) 
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], shape).round()

            # write results to string
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # pass detections to strongsort
            with torch.no_grad():
                with dt[3]:
                    outputs[i] = tracker_list[i].update(det.cpu(), im0)

            # draw boxes for visualization
            if len(outputs[i]) > 0:
                
                retina_masks = False
                annotator.masks(
                    masks[i],
                    colors=[colors(x, True) for x in det[:, 5]],
                    im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(torch.device('cpu')).permute(2, 0, 1).flip(0).contiguous() /
                    255 if retina_masks else im[i]
                )

                # iterate over detected objects
                for j, (output) in enumerate(outputs[i]):
                    
                    bbox = output[0:4]
                    id = output[4]
                    cls = output[5]
                    conf = output[6]

                    face_check = False

                    # if this is a person and we havent checked their face yet, check it
                    if names[int(output[5])] == 'person' and int(output[4]) not in checked_people:
                        if not face_check:
                            face_locations = face_recognition.face_locations(frame)
                            face_check = True
                            # iterate over detected faces, if the location overlaps with the person, encode and compare
                            for (top, right, bottom, left) in face_locations:
                                if (output[0] < left < output[2]) and (output[1] < top < output[3]):
                                    face_encodings = face_recognition.face_encodings(frame, face_locations)
                                    face_names = []
                                    # match encodings to known encodings
                                    for face_encoding in face_encodings:
                                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                                        name = "Unknown"
                                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                                        if face_distances is not None:
                                            best_match_index = np.argmin(face_distances)
                                            if matches[best_match_index]:
                                                name = known_face_names[best_match_index]
                                        face_names.append(name)
                                    checked_people[int(output[4])] = name
                                    break

                    # if this is a person we want to follow, follow them
                    if names[int(cls)] == 'person' and id in checked_people and checked_people[id] == target_person:
                        pixels = bbox[2] - bbox[0]
                        distance = utils.depth_estimation(follow_data['known_width'], follow_data['focal_length'], pixels)
                        target_node = utils.find_closest_node(bbox, distance, follow_data['rows'], follow_data['cols'], follow_data['frame_width'], follow_data['focal_length'])
                        path, straight_shot_node = utils.dijkstras(follow_data['adjacency_matrix'], follow_data['start_node'], target_node, follow_data['cols'])
                        rotation_angle, euclidean_distance = utils.determine_commands(straight_shot_node, follow_data['focal_length'], follow_data['cols'], follow_data['start_node'])

                    # generate visualizations
                    c = int(cls) 
                    id = int(id)
                    
                    # if c is a person, check if we have a name for them
                    if names[c] == 'person' and id in checked_people:
                        name = checked_people[id]
                    else:
                        name = names[c]
                    hide_labels = False
                    hide_class = False
                    hide_conf = False
                    label = None if hide_labels else (f'{id} {name}' if hide_conf else \
                        (f'{id} {conf:.2f}' if hide_class else f'{id} {name} {conf:.2f}'))
                    color = colors(c, True)
                    annotator.box_label(bbox, label, color=color)
        else:
            pass
            
        # Stream results
        im0 = annotator.result()
        cv2.imshow(str(p), im0)
        if cv2.waitKey(1) == ord('q'):  # 1 millisecond
            exit()
        prev_frame = curr_frames[i]
    return prev_frame, rotation_angle, euclidean_distance