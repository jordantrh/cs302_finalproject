from ultralytics.nn.autobackend import AutoBackend
# from ultralytics.yolo.utils.checks import check_imgsz, check_imshow
from ultralytics.yolo.utils.ops import Profile, non_max_suppression, scale_boxes, process_mask
from ultralytics.yolo.utils.plotting import Annotator, colors
from ultralytics.yolo.data.augment import LetterBox

import numpy as np
import torch
import cv2
from pathlib import Path
import face_recognition
# from sys import platform

import utils

def transform(im0, imgsz, pt, stride):
    im = np.stack([LetterBox(imgsz, pt, stride=stride)(image=x) for x in im0])
    im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
    im = np.ascontiguousarray(im)  # contiguous
    return im 

def track(frame, model, tracker_list, checked_people, known_face_encodings, known_face_names, follow_data, target_person):
    conf_thres = 0.25
    iou_thres = 0.45
    max_det = 1000
    classes = None
    agnostic_nms = False
    imgsz = (640, 640)
    bs = 1
    outputs = [None] * bs
    curr_frames, prev_frames = [None] * bs, [None] * bs
    stride, names, pt = model.stride, model.names, model.pt
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile(), Profile())
    im = transform([frame], imgsz, pt, stride)
    im0s = [frame]
    s = ''
    rotation_angle = None
    euclidean_distance = None

    with dt[0]:
        im = torch.from_numpy(im).to(torch.device('cpu'))
        im = im.half() if False else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

    # Inference
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

        s += '%gx%g ' % im.shape[2:]  # print string

        annotator = Annotator(im0, line_width=2, example=str(names))
        
        if hasattr(tracker_list[i], 'tracker') and hasattr(tracker_list[i].tracker, 'camera_update'):
            if prev_frames[i] is not None and curr_frames[i] is not None:  # camera motion compensation
                tracker_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

        if det is not None and len(det):
            shape = im0.shape
            # scale bbox first the crop masks
            masks.append(process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True))  # HWC
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], shape).round()  # rescale boxes to im0 size

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
                
                retina_masks = False
                annotator.masks(
                    masks[i],
                    colors=[colors(x, True) for x in det[:, 5]],
                    im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(torch.device('cpu')).permute(2, 0, 1).flip(0).contiguous() /
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
                    if names[int(cls)] == 'person' and id in checked_people and checked_people[id] == target_person:
                        pixels = bbox[2] - bbox[0]
                        distance = utils.depth_estimation(follow_data['known_width'], follow_data['focal_length'], pixels)
                        tracking = True
                        if tracking:
                            target_node = utils.find_closest_node(bbox, distance, follow_data['rows'], follow_data['cols'], follow_data['frame_width'], follow_data['focal_length'])
                            path, straight_shot_node = utils.dijkstras(follow_data['adjacency_matrix'], follow_data['start_node'], target_node, follow_data['cols'])
                            rotation_angle, euclidean_distance = utils.determine_commands(straight_shot_node, follow_data['focal_length'], follow_data['cols'], follow_data['start_node'])

                    c = int(cls)  # integer class
                    id = int(id)  # integer id
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
        # if platform.system() == 'Linux' and p not in windows:
        #     windows.append(p)
        #     cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
        #     cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
        cv2.imshow(str(p), im0)
        if cv2.waitKey(1) == ord('q'):  # 1 millisecond
            exit()
    return rotation_angle, euclidean_distance