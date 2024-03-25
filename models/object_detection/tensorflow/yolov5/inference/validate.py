#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Intel Corporation
#
# AGPL-3.0 license
#

# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license

import argparse
import os
import glob
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
import shutil

FILE = Path(__file__).resolve()
ROOT = FILE.parent  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
DETECT_DIR = FILE.parent
if str(DETECT_DIR) not in sys.path:
    sys.path.insert(0, str(DETECT_DIR))

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
<<<<<<< HEAD
=======
from data.unpack_json_labels import unpack_json_labels
>>>>>>> be0e7cd1b ([Tensorflow] Enable bfloat16, fp16, and int8 for Yolo V5 (#1790))
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    Profile,
    check_dataset,
    check_img_size,
    check_requirements,
    check_yaml,
    coco80_to_coco91_class,
    colorstr,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    xywh2xyxy,
)
from utils.metrics import ap_per_class, box_iou
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, smart_inference_mode

def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix.

    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)

@smart_inference_mode()
def run(
    yaml_path,
    precision,
    source=ROOT / "../data/images/val2017",
    weights=None,  # model.pt path(s)
    imgsz=640,  # inference size (pixels)
    conf_thres=0.001,  # confidence threshold
    iou_thres=0.6,  # NMS IoU threshold
    max_det=300,  # maximum detections per image
    task="val",  # train, val, test, speed or study
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    workers=8,  # max dataloader workers (per RANK in DDP mode)
    single_cls=False,  # treat as single-class dataset
    verbose=False,  # verbose output
    project=ROOT / "runs/val",  # save to project/name
    name="exp",  # save to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    model=None,
    dataloader=None,
    save_dir=Path(""),
    callbacks=Callbacks(),
    compute_loss=None,
<<<<<<< HEAD
=======
    instances_json = None,
>>>>>>> be0e7cd1b ([Tensorflow] Enable bfloat16, fp16, and int8 for Yolo V5 (#1790))
    speed = False,
):
    device = select_device(device)

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels").mkdir(parents=True, exist_ok=True)  # make dir

<<<<<<< HEAD
=======
    # Check if source is a real directory
    assert os.path.isdir(source), "Source is not a real directory"

    # Check if the directory is not empty
    assert os.listdir(source), "Directory is empty"

    # Check if all files in the directory end with '.jpg'
    all_jpg = all(file.endswith('.jpg') for file in glob.glob(os.path.join(source, '*')))
    assert all_jpg, "Not all files in the directory end with .jpg"

    print("Directory is valid and all files end with .jpg")

    # If data is not in data/images/val2017
    if source != ROOT / "data/images/val2017":
        destination = ROOT / "data/images/val2017"
        os.makedirs(destination, exist_ok=True)
        for file_name in os.listdir(source):
            full_file_name = os.path.join(source, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, destination)
                copied_file_path = os.path.join(destination, file_name)
                os.chmod(copied_file_path, 0o644)
    
    if instances_json:
        unpack_json_labels(instances_json, ROOT)

>>>>>>> be0e7cd1b ([Tensorflow] Enable bfloat16, fp16, and int8 for Yolo V5 (#1790))
    # Load model
    model = DetectMultiBackend(weights, precision=precision, device=device, data=yaml_path)
    stride, pt = model.stride, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    device = model.device
    batch_size = 1  # export.py models default to batch-size 1
    LOGGER.info(f"Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models")
    
    # Data
    data = check_dataset(yaml_path)  # check

    # Configure
    model.eval()
    nc = 1 if single_cls else int(data["nc"])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
    pad, rect = (0.5, pt)  # square inference for benchmarks

    dataloader = create_dataloader(
        data[task],
        imgsz,
        batch_size,
        stride,
        single_cls,
        pad=pad,
        rect=rect,
        workers=workers,
        prefix=colorstr(f"{task}: "),
    )[0]

    seen = 0
    names = model.names if hasattr(model, "names") else model.module.names  # get class names
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))
    s = ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "P", "R", "mAP50", "mAP50-95")
    p, r, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dt = Profile(), Profile(), Profile()  # profiling times
    loss = torch.zeros(3, device=device)
    stats, ap, ap_class = [], [], []
    callbacks.run("on_val_start")
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        callbacks.run("on_val_batch_start")
        with dt[0]:
            im = im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            _, _, height, width = im.shape  # batch size, channels, height, width

        # Inference
        with dt[1]:
            preds, train_out = model(im) if compute_loss else (model(im, augment=False), None)
        
        # Batch size 1 inference drops the batch dim
        if preds.dim() == 2:
            preds=preds.unsqueeze(0)

        # Loss
        if compute_loss:
            loss += compute_loss(train_out, targets)[1]  # box, obj, cls

        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        lb = []  # for autolabelling
        with dt[2]:
            preds = non_max_suppression(
                preds, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls, max_det=max_det
            )

        # Metrics
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

            callbacks.run("on_val_image_end", pred, predn, path, names, im[si])


        callbacks.run("on_val_batch_end", batch_i, im, targets, paths, shapes, preds)

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        _, _, p, r, _, ap, ap_class = ap_per_class(*stats, plot=False, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class

    # Print results
<<<<<<< HEAD
    pf = "%22s" + "%11i" * 2 + "%11.4g" * 4  # print format
    # LOGGER.info(pf % ("all", seen, nt.sum(), mp, mr, map50, map))
    LOGGER.info("Accuracy (map50): %.4g" % map50)
=======
    pf = "%22s" + "%11i" * 2 + "%11.3g" * 4  # print format
    # LOGGER.info(pf % ("all", seen, nt.sum(), mp, mr, map50, map))
    LOGGER.info("Accuracy (map50): %.3g" % map50)
>>>>>>> be0e7cd1b ([Tensorflow] Enable bfloat16, fp16, and int8 for Yolo V5 (#1790))
    if nt.sum() == 0:
        LOGGER.warning(f"WARNING ⚠️ no labels found in {task} set, can not compute metrics without labels")

    # Print results per class
    if (verbose or (nc < 50)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    shape = (batch_size, 3, imgsz, imgsz)
    if speed:
        LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}" % t)

    # Return results
    model.float()  # for training
    # LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", type=str, default="fp32", help="precision i.e. fp32, fp16, bfloat16, int8")
<<<<<<< HEAD
    parser.add_argument("--source", type=str, default=ROOT / "datasets/coco/images", help="path to coco validation dataset")
    parser.add_argument("--yaml-path", default=ROOT / "data/coco.yaml", help="dataset.yaml path")
=======
    parser.add_argument("--source", type=str, default=ROOT / "data/images/val2017", help="path to coco validation dataset")
    parser.add_argument("--yaml-path", default=ROOT / "data/coco-val.yaml", help="dataset.yaml path")
>>>>>>> be0e7cd1b ([Tensorflow] Enable bfloat16, fp16, and int8 for Yolo V5 (#1790))
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pb", help="model path(s)")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--single-cls", action="store_true", help="treat as single-class dataset")
    parser.add_argument("--verbose", action="store_true", help="report mAP by class")
    parser.add_argument("--project", default=ROOT / "runs/val", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
<<<<<<< HEAD
=======
    parser.add_argument("--instances-json", default=None, type=str, help="Path to the coco-val2017-instances json file")
>>>>>>> be0e7cd1b ([Tensorflow] Enable bfloat16, fp16, and int8 for Yolo V5 (#1790))
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
