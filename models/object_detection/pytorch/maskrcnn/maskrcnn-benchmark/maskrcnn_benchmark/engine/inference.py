#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os

import torch
from tqdm import tqdm

from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
from .bbox_aug import im_detect_bbox_aug
import intel_extension_for_pytorch as ipex
# from maskrcnn_benchmark.engine.utils_vis import draw, make_dot


def compute_on_dataset(model, data_loader, device, bbox_aug, timer=None, bf16=False, jit=False, iterations=-1, iter_warmup=-1, enable_profiling=False):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    steps_per_epoch = len(data_loader)
    iter_warmup = max(0, iter_warmup)
    total_steps = (iterations if iterations > 0 else steps_per_epoch) + iter_warmup
    test_epoches = int(total_steps / steps_per_epoch)
    print('Evaluating MaskRCNN: Steps per Epoch {} total Steps {}'.format(steps_per_epoch, total_steps))

    model = model.to(memory_format=torch.channels_last)
    model.backbone = ipex.optimize(model.backbone, dtype=torch.bfloat16 if bf16 else torch.float, inplace=True)
    model.rpn = ipex.optimize(model.rpn, dtype=torch.bfloat16 if bf16 else torch.float, inplace=True)
    model.roi_heads = ipex.optimize(model.roi_heads, dtype=torch.bfloat16 if bf16 else torch.float, inplace=True)

    with torch.cpu.amp.autocast(enabled=bf16), torch.no_grad():
        # generate trace model
        if jit:
            print("generate trace model")
            for i, batch in enumerate(tqdm(data_loader)):
                images, targets, image_ids = batch
                model.backbone = torch.jit.trace(model.backbone, images.tensors.to(memory_format=torch.channels_last))
                model.backbone = torch.jit.freeze(model.backbone)
                trace_graph = model.backbone.graph_for(images.tensors.to(memory_format=torch.channels_last))
                print(trace_graph)
                break
        # Inference
        print("runing inference step")
        with torch.autograd.profiler.profile(enable_profiling) as prof:
            with tqdm(total=total_steps, desc="Evaluating") as pbar:
                for epoch in range(test_epoches + 1):
                    for i, batch in enumerate(data_loader):
                        if epoch * steps_per_epoch + i >= total_steps:
                            break
                        images, targets, image_ids = batch
                        images = images.to(memory_format=torch.channels_last)
                        
                        if bf16:
                            images = images.to(torch.bfloat16)
                        if timer and epoch * steps_per_epoch + i >= iter_warmup:
                            timer.tic()
                        if bbox_aug:
                            output = im_detect_bbox_aug(model, images, device)
                        else:
                            output = model(images)
                        if timer and epoch * steps_per_epoch + i >= iter_warmup:
                            timer.toc()
                        output = [o.to(cpu_device) for o in output]
                        results_dict.update(
                            {img_id: result for img_id, result in zip(image_ids, output)}
                        )
                        pbar.update(1)
        if enable_profiling:
            print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        model,
        data_loader,
        dataset_name,
        ims_per_patch,
        iou_types=("bbox",),
        box_only=False,
        bbox_aug=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
        bf16=False,
        jit=False,
        iterations=-1,
        iter_warmup=-1,
        enable_profiling=False
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader, device, bbox_aug, inference_timer, bf16, jit, iterations, iter_warmup, enable_profiling)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)

    if iterations == -1:
        iterations = len(data_loader)

    logger.info(
        "Total run time: {} ({} s / iter per device, on {} devices)".format(
            total_time_str, total_time * num_devices / iterations, num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / iter per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / iterations,
            num_devices,
        )
    )

    print("Throughput: {:.3f} fps".format((iterations * ims_per_patch) / (inference_timer.total_time * num_devices)))  

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)
