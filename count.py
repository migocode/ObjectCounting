import argparse

import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov7') not in sys.path:
    sys.path.append(str(ROOT / 'yolov7'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

print(sys.path)

from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadImages, LoadStreams
from yolov7.utils.general import (check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, strip_optimizer, check_file)
from yolov7.utils.torch_utils import select_device
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT
from Threshold import Threshold, Line, Point
from OverlayPlotter import OverlayPlotter
from ObjectCounter import ObjectCounter
from MessageQueue import MessageQueue


VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes


@torch.no_grad()
def run(
        source='0',
        yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        strong_sort_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        config_strongsort=ROOT / 'strong_sort/configs/strong_sort.yaml',
        imgsz=[640, 640],  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, source_index.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_vid=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    image_size = imgsz

    yolo_weights = Path([yolo_weights][0])

    # Load model
    device = select_device(device)

    WEIGHTS.mkdir(parents=True, exist_ok=True)
    model = attempt_load(Path(yolo_weights), map_location=device)  # load FP32 model
    names, = model.names,
    stride = model.stride.max()  # model stride
    image_size = check_img_size(image_size[0], s=stride.cpu().numpy())  # check image size

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=image_size, stride=stride.cpu().numpy())
        nr_sources = 1
    else:
        dataset = LoadImages(source, img_size=image_size, stride=stride)
        nr_sources = 1

    # initialize StrongSORT
    cfg = get_config()
    cfg.merge_from_file(opt.config_strongsort)

    threshold_start = Point(int(640/2), 0)
    threshold_end = Point(int(640/2), 480)

    # Create as many strong sort & object counter instances as there are video sources
    strongsort_list = []
    object_counter_list = []
    for source_index in range(nr_sources):
        strongsort_list.append(
            StrongSORT(
                strong_sort_weights,
                device,
                half,
                max_dist=cfg.STRONGSORT.MAX_DIST,
                max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                max_age=cfg.STRONGSORT.MAX_AGE,
                n_init=cfg.STRONGSORT.N_INIT,
                nn_budget=cfg.STRONGSORT.NN_BUDGET,
                mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                ema_alpha=cfg.STRONGSORT.EMA_ALPHA,
            )
        )
        strongsort_list[source_index].model.warmup()

        object_counter_list.append(
            ObjectCounter(
                thresholds=[Threshold(threshold_line=Line(threshold_start, threshold_end), outside_direction=1)],
                max_age=cfg.STRONGSORT.MAX_AGE
            )
        )

    strong_sort_outputs = [None] * nr_sources

    overlay_plotter = OverlayPlotter(1, 2, names)

    # Message Queue
    message_queue: MessageQueue = MessageQueue("passenger_count")

    # Run tracking
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
    for frame_idx, (path, im, im0s, vid_cap) in enumerate(dataset):
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        yolo_detections = model(im)

        # Apply NMS
        yolo_detections = non_max_suppression(yolo_detections[0], conf_thres, iou_thres, classes, agnostic_nms)

        # Process detections per source (source index: source_index)
        for source_index, detections in enumerate(yolo_detections):  # detections per image
            seen += 1

            strong_sort = strongsort_list[source_index]
            strongsort_output = strong_sort_outputs[source_index]
            object_counter = object_counter_list[source_index]

            p, im0, _ = path[source_index], im0s[source_index].copy(), dataset.count
            p = Path(p)  # to Path

            overlay_plotter.plot_path(threshold_start.get_tuple(),
                                      threshold_end.get_tuple(), im0)

            curr_frames[source_index] = im0

            if cfg.STRONGSORT.ECC:  # camera motion compensation
                strong_sort.tracker.camera_update(prev_frames[source_index], curr_frames[source_index])

            if detections is not None and len(detections):
                # Rescale boxes from img_size to im0 size
                detections[:, :4] = scale_coords(im.shape[2:], detections[:, :4], im0.shape).round()

                boxes_xywh = xyxy2xywh(detections[:, 0:4])
                confidence_scores = detections[:, 4]
                classes = detections[:, 5]

                # Perform tracking
                strongsort_output = strong_sort.update(boxes_xywh.cpu(),
                                                       confidence_scores.cpu(),
                                                       classes.cpu(),
                                                       im0)

                # draw boxes for visualization
                if len(strongsort_output) > 0:
                    for j, (output, confidence_score) in enumerate(zip(strongsort_output, confidence_scores)):
                        confidence_score = float(confidence_score.cpu().numpy())

                        bounding_box = output[:4]
                        xy = bounding_box[:2]
                        wh = bounding_box[2:]
                        id_detection = output[4]
                        predicted_class = output[5]

                        center = Threshold.get_box_center(xy, wh)
                        direction_metric = object_counter.check(id_detection, predicted_class, center)

                        # Convert direction metric to -1, 0, 1 for counting
                        direction = direction_metric / abs(direction_metric) if direction_metric != 0 else 0

                        class_id = int(predicted_class)  # integer class
                        class_name = names[class_id]
                        id_detection = int(id_detection)  # integer id

                        if direction != 0:
                            print(f"Object id: {id_detection}")
                            print(f"Class id: {class_id}")
                            print(f"Class name: {class_name}")
                            print(f"Direction: {direction}")
                            print(f"Tracked point {center}")
                            print("---------------------")

                            message_queue.publish(id_detection,
                                                  class_id,
                                                  class_name,
                                                  direction,
                                                  confidence_score)

                        overlay_plotter.plot_one_box(bounding_box,
                                                     center=(center.x, center.y),
                                                     img=im0,
                                                     class_id=class_id,
                                                     id_detection=id_detection,
                                                     confidence_score=confidence_score)

            else:
                strongsort_list[source_index].increment_ages()
                object_counter_list[source_index].increment_ages()

            # Stream results
            if show_vid:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            prev_frames[source_index] = curr_frames[source_index]

    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)

    del message_queue

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=str, default=WEIGHTS / 'yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--strong-sort-weights', type=str, default=WEIGHTS / 'osnet_x0_25_market_256x128_amsgrad_ep180_stp80_lr0.003_b128_fb10_softmax_labelsmooth_flip.pt')
    parser.add_argument('--config-strongsort', type=str, default='strong_sort/configs/strong_sort.yaml')
    parser.add_argument('--source', type=str, default='rtsp://10.235.156.102:8554/cam', help='file/dir/URL/glob, 0 for webcam')
    #parser.add_argument('--source', type=str, default='1', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640, 640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements_backup.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
