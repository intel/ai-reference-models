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
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
DETECT_DIR = FILE.parent.parent
if str(DETECT_DIR) not in sys.path:
    sys.path.insert(0, str(DETECT_DIR))

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImagesAndLabels
from utils.general import (LOGGER, Profile, check_img_size, check_requirements, non_max_suppression, print_args)
from utils.torch_utils import select_device, smart_inference_mode

@smart_inference_mode()
def run(
        weights,  # model path
        source,  # path to dataset
        precision,  # precision
        bs = 1,  # batched inference
        imgsz=640,  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
):
    source = str(source)

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, precision=precision, device=device)
    stride, names = model.stride, model.names
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    dataset = LoadImagesAndLabels(source, img_size=imgsz, stride=stride, batch_size=bs)
    total_images = dataset.n
    data_loader = DataLoader(dataset, batch_size=bs, collate_fn=LoadImagesAndLabels.collate_fn)

    # Run inference
    #model.warmup(imgsz=(bs, 3, *imgsz))  # warmup
    seen = 0
    dt = (Profile(), Profile(), Profile())

    b_count = 0

    for batch in data_loader:
        b_count += 1
        if seen + bs >= total_images:
            break
        im, labels, paths, shapes = batch
        out_strs = ['image ']*bs

        # Image Preprocessing
        with dt[0]:
            im = im.to(model.device).float()
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
        
        # Inference
        with dt[1]:
            pred = model(im)
        
        if type(pred)==list:
            pred = torch.stack(pred, dim=0)
        
        # Batch size 1 inference drops the batch dim
        if pred.dim() == 2:
            pred=pred.unsqueeze(0)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        
        # Process Predictions
        for i, det in enumerate(pred):  # per image in batch
            seen += 1
            out_strs[i] +=  f'{seen}/{total_images} '
            out_strs[i] += '%gx%g ' % im.shape[2:]  # print image shape

            if len(det):  # If objects detected
                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # number of detections per class
                    out_strs[i] += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to output string
            
            LOGGER.info(f"{out_strs[i]}{'' if len(det) else '(no detections)'}")
        
        LOGGER.info(f"Batch {b_count} time: {dt[1].dt * 1E3:.1f}ms")

    time_per_img = tuple((x.t / seen * 1E3) for x in dt)
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, imgsz, imgsz)}' % time_per_img)
    if bs==1:
        LOGGER.info(f'Latency: {time_per_img[1]:.1f}ms')
        LOGGER.info(f'Avg Throughput: {1000 / time_per_img[1]:.1f} images/s')
    else:
        LOGGER.info(f'Avg Throughput: {1000 / time_per_img[1]:.1f} images/s')

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, help='model path')
    parser.add_argument('--source', type=str, help='path to dataset')
    parser.add_argument('--bs', type=int, default=1, help='batch size')
    parser.add_argument('--precision', default='fp32', help='precision: fp32, bfloat16, fp16, int8')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
