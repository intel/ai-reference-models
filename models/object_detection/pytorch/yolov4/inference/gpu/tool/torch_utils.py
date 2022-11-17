import sys
import os
import time
import math
import torch
import numpy as np
from torch.autograd import Variable

import itertools
import struct  # get_image_size
import imghdr  # get_image_size

from tool import utils 

def bbox_ious(boxes1, boxes2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = torch.min(boxes1[0], boxes2[0])
        Mx = torch.max(boxes1[2], boxes2[2])
        my = torch.min(boxes1[1], boxes2[1])
        My = torch.max(boxes1[3], boxes2[3])
        w1 = boxes1[2] - boxes1[0]
        h1 = boxes1[3] - boxes1[1]
        w2 = boxes2[2] - boxes2[0]
        h2 = boxes2[3] - boxes2[1]
    else:
        mx = torch.min(boxes1[0] - boxes1[2] / 2.0, boxes2[0] - boxes2[2] / 2.0)
        Mx = torch.max(boxes1[0] + boxes1[2] / 2.0, boxes2[0] + boxes2[2] / 2.0)
        my = torch.min(boxes1[1] - boxes1[3] / 2.0, boxes2[1] - boxes2[3] / 2.0)
        My = torch.max(boxes1[1] + boxes1[3] / 2.0, boxes2[1] + boxes2[3] / 2.0)
        w1 = boxes1[2]
        h1 = boxes1[3]
        w2 = boxes2[2]
        h2 = boxes2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    mask = ((cw <= 0) + (ch <= 0) > 0)
    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    carea[mask] = 0
    uarea = area1 + area2 - carea
    return carea / uarea


def get_region_boxes(boxes_and_confs):

    # print('Getting boxes from boxes and confs ...')

    boxes_list = []
    confs_list = []

    for item in boxes_and_confs:
        boxes_list.append(item[0])
        confs_list.append(item[1])

    # boxes: [batch, num1 + num2 + num3, 1, 4]
    # confs: [batch, num1 + num2 + num3, num_classes]
    boxes = torch.cat(boxes_list, dim=1)
    confs = torch.cat(confs_list, dim=1)
        
    return [boxes, confs]


def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)


def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)



def do_detect(model, img, conf_thresh, nms_thresh, use_cuda=0, iteration=0, dummy=0, batch_size=1, width=0, height=0, data_type="", benchmark=0):
    model.eval()
    t0 = time.time()
    if dummy == 0:
        if type(img) == np.ndarray and len(img.shape) == 3:  # cv2 image
            img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
        elif type(img) == np.ndarray and len(img.shape) == 4:
            img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float().div(255.0)
        else:
            print("unknow image type")
            exit(-1)

        if use_cuda:
            img = img.cuda()
    else:
        img = torch.randn(batch_size, 3, width, height, requires_grad=True)

    profiling = os.environ.get("PROFILE", "OFF").upper() in ["1", "Y", "ON", "YES", "TRUE"]
    if benchmark == 1:
        img = img.to("xpu")
    with torch.autograd.profiler_legacy.profile(enabled=profiling, use_xpu=True, record_shapes=False) as prof:
        t1 = time.time()
        if benchmark == 0:
            img = img.to("xpu")
        if data_type == "fp16":
            img = img.half()
        img = torch.autograd.Variable(img)
        output = model(img)

        # sync for time measurement
        torch.xpu.synchronize()
        if benchmark == 1:
            inf_latency = time.time() - t1
    ret = None
    if benchmark == 0:
        output = [x.to("cpu") for x in output]
        ret = utils.post_processing(img, conf_thresh, nms_thresh, output)
        inf_latency = time.time() - t1

    process_latency = t1 - t0
    if iteration > 0:
        print('-----------------------------------')
        print('           Preprocess : %f' % process_latency)
        print('      Model Inference : %f' % inf_latency)
        print('-----------------------------------')

    if profiling:
        title = "/yolov4_inference_"
        title += data_type + "_"
        title += "bs" + str(batch_size) + "_"

        profiling_path = os.getenv('PROFILE_PATH')
        if not profiling_path:
            profiling_path = './'
        torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"), profiling_path + title + 'profiling.pt')
        torch.save(prof.key_averages(group_by_input_shape=True).table(), profiling_path + title + 'profiling_detailed.pt')
        prof.export_chrome_trace(profiling_path + title + 'profiling.json')
        print(prof.key_averages().table(sort_by="self_xpu_time_total"))
        print(prof.key_averages(group_by_input_shape=True).table())

    return ret, inf_latency

