#
# Copyright (c) 2021 Intel Corporation
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

import argparse
import time

import torch
import torchvision
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import DataLoader

from coco_eval import CocoEvaluator
from presets import DetectionPresetEval
from coco_utils import get_coco_api_from_dataset, get_coco
from utils import collate_fn
from retinanet import *

model_names = sorted(name for name in torchvision.models.detection.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision.models.detection.__dict__[name]))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def inference(model, dataloader, datatype, args):
    batch_time = AverageMeter('Time', ':6.3f')
    batch_size = args.batch_size
    warmup_iters = args.warmup_iterations
    max_iters = args.max_iterations if dataloader is None else len(dataloader)
    model.eval()
    coco = get_coco_api_from_dataset(dataloader.dataset)
    iou_types = ["bbox"]
    coco_evaluator = CocoEvaluator(coco, iou_types)
    if args.ipex:
        import intel_extension_for_pytorch as ipex
        model = model.to(memory_format=torch.channels_last)
        model = ipex.optimize(model, dtype=datatype, level="O1", conv_bn_folding=False, replace_dropout_with_identity=False)
        model.backbone = ipex.optimize(model.backbone, dtype=datatype, level="O1")
    else:
        if args.jit:
            model = model.to(memory_format=torch.channels_last)
        else:
            from torch.utils import mkldnn as mkldnn_utils
            model = mkldnn_utils.to_mkldnn(model, dtype=datatype)
    if args.jit:
        x = torch.randn(batch_size, 3, 1200, 1200).to(memory_format=torch.channels_last)
        if args.precision == "bf16":
            with torch.cpu.amp.autocast(), torch.no_grad():
                model.backbone = torch.jit.trace(model.backbone, x, strict=False)
            model.backbone = torch.jit.freeze(model.backbone)
        else:
            with torch.no_grad():
                model.backbone = torch.jit.trace(model.backbone, x, strict=False)
            model.backbone = torch.jit.freeze(model.backbone)
    with torch.no_grad():
        if dataloader is None:
            print("Models for detection tasks need to use real dataset. You need to specify coco dataset. ")
            exit(1)
        else:
            for i, batch in enumerate(dataloader):
                images = batch[0]
                if not args.ipex and not args.jit:
                    images = list(img.to(datatype) for img in images)
                if args.ipex and args.precision == "bf16":
                    with torch.cpu.amp.autocast():
                        if i == warmup_iters:
                            with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof, record_function("model_inference"):
                                output = model(images)
                        else:
                            output = model(images)
                else:
                    if i == warmup_iters:
                        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof, record_function("model_inference"):
                            output = model(images)
                    else:
                        output = model(images)
                if i > warmup_iters:
                    break
            for i, batch in enumerate(dataloader):
                images = batch[0]
                end = time.time()
                if not args.ipex and not args.jit:
                    images = list(img.to(datatype) for img in images)
                if args.ipex and args.precision == "bf16":
                    with torch.cpu.amp.autocast():
                        output = model(images)
                else:
                    output = model(images)
                batch_time.update(time.time() - end)
                output = [{k: v.to(torch.float32) for k, v in t.items()} for t in output]
                res = {target["image_id"].item(): output for target, output in zip(batch[1], output)}
                coco_evaluator.update(res)
                if max_iters != -1 and i >= max_iters:
                    break
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=-1))
    latency = batch_time.avg / batch_size * 1000
    perf = batch_size / batch_time.avg
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    print("Bbox AP: {:.5f} ".format(coco_evaluator.coco_eval['bbox'].stats[0]))
    print('Latency: %.3f ms'%latency)
    print("Throughput: {:.3f} fps".format(perf))

def get_coco_data_loader(coco_image_dir, batch_size):
    transforms = DetectionPresetEval()
    coco_test_data = get_coco(root=coco_image_dir, image_set="val", transforms=transforms)
    coco_test_data_loader = DataLoader(coco_test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return coco_test_data_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch models inference')
    parser.add_argument('--data_path', '-d', type=str, default=None,
                        help='path to dataset (default: None)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='fasterrcnn_resnet50_fpn',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: fasterrcnn_resnet50_fpn)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch_size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('-p', '--print_freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', default=True,
                        help='use pre-trained model')

    parser.add_argument('--ipex', action='store_true', default=False,
                        help='use intel pytorch extension')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disable CUDA')
    parser.add_argument('--precision', default='fp32', type=str,
                        help='set precision of running model, can be fp32, bf16 or int8 (todo)')
    parser.add_argument('--jit', action='store_true', default=False,
                        help='enable jit fusion path')
    parser.add_argument("--dummy", action='store_true',
                        help="using  dummu data to test the performance of inference")
    parser.add_argument('-w', '--warmup_iterations', default=30, type=int, metavar='N',
                        help='number of warmup iterations to run')
    parser.add_argument('-m', '--max_iterations', default=50, type=int,
                        help='number of max iterations to run')
    parser.add_argument('--log-path', required = False, default = "", type=str,
                        help="Path for the log file")
    parser.add_argument("--cache-dataset", dest="cache_dataset", default=False, action="store_true",
                        help='Cache the datasets for quicker initialization. '
                            'It also serializes the transforms')
    args = parser.parse_args()
    model = retinanet_resnet50_fpn(pretrained=args.pretrained)
    batch_size = args.batch_size
    datatype=torch.float32
    dataset_dir = args.data_path
    if args.precision == 'bf16':
        datatype=torch.bfloat16
    dataloader = get_coco_data_loader(dataset_dir, batch_size)
    inference(model, dataloader, datatype, args)
