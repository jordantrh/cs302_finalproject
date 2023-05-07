import cv2
import os
import pdb
import sys
import platform
import numpy as np
from pathlib import Path
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT / 'trackers' / 'strongsort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strongsort'))  # add strong_sort ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.utils.checks import check_imgsz, check_imshow
from ultralytics.yolo.utils.ops import Profile, non_max_suppression, scale_boxes, process_mask, process_mask_native
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

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

from ultralytics.yolo.data.augment import LetterBox
def transform(im0):
    im = np.stack([LetterBox(imgsz, pt, stride=stride)(image=x) for x in im0])
    im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
    im = np.ascontiguousarray(im)  # contiguous
    return im 

video_capture = cv2.VideoCapture(0)
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

                    # TODO add distance estimation
                    '''if names[cls] == 'person':
                        focal_length = 52
                        height = 1066 
                        known_size = abs(height-1737) # height - 5'7"
                        pixels = abs(bbox[3] - im0.shape[0])
                        distance = known_size * focal_length / pixels

                        distance = distance * 0.0393701
                        print(f'{distance/12} feet away')'''

                    if show_vid:
                        c = int(cls)  # integer class
                        id = int(id)  # integer id
                        label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                            (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
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