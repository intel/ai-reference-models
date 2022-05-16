# Copyright (c) 2022 Intel Corporation
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

import os
import os.path
import time
import torch
import numpy as np

import argparse
from transnetv2 import TransNetV2
from torch.profiler import profile, ProfilerActivity

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

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

def get_batches(frames):
    length = frames.shape[0]
    reminder = 50 - length % 50
    if reminder == 50:
        reminder = 0
    frames = torch.cat([frames[:1]] * 25 + [frames] + [frames[-1:]] * (reminder + 25), 0)

    length = frames.shape[0]
    def func():
        for i in range(0, length - 50, 50):
            yield frames[i:i+100]
    return func()

def inference(model, dataloader, args):
    batch_time = AverageMeter('Time', ':6.3f')
    batch_size = args.batch_size
    total_stats = {"tp": 0, "fp": 0, "fn": 0}
    warmup_iters = args.warmup_iterations
    max_iters = args.max_iterations
    logit_fc = torch.sigmoid
    datatype=torch.float32
    if args.precision == 'bf16':
        datatype=torch.bfloat16
    model.eval()

    if args.ipex:
        import intel_extension_for_pytorch as ipex
        model = model.to(memory_format=torch.channels_last_3d)
        model = ipex.optimize(model, dtype=datatype, inplace=True)
    else:
        if args.jit:
            model = model.to(memory_format=torch.channels_last_3d)
        else:
            from torch.utils import mkldnn as mkldnn_utils
            model = mkldnn_utils.to_mkldnn(model, dtype=datatype)
    if args.jit:
        x = torch.randint(0, 255, (args.batch_size, 100, 27, 48, 3), dtype=datatype).to(memory_format=torch.channels_last_3d)
        if args.precision == "bf16":
            with torch.cpu.amp.autocast(), torch.no_grad():
                model = torch.jit.trace(model, x, strict=False).eval()
            model = torch.jit.freeze(model)
        else:
            with torch.no_grad():
                model = torch.jit.trace(model, x, strict=False).eval()
            model = torch.jit.freeze(model)

    with torch.no_grad():
        for i in range(warmup_iters+1):
            images = torch.randint(0, 255, (args.batch_size, 100, 27, 48, 3), dtype=datatype).to(memory_format=torch.channels_last_3d)
            if args.ipex and args.precision == "bf16":
                with torch.cpu.amp.autocast():
                    if i == warmup_iters:
                        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                            output = model(images)
                    else:
                        output = model(images)
            else:
                if i == warmup_iters:
                    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                        output = model(images)
                else:
                    output = model(images)

        for i in range(max_iters):
            images = torch.randint(0, 255, (args.batch_size, 100, 27, 48, 3))
            end = time.time()
            images = images.to(datatype).to(memory_format=torch.channels_last_3d)
            if args.ipex and args.precision == "bf16":
                with torch.cpu.amp.autocast():
                    output = model(images)
            else:
                output = model(images)
            batch_time.update(time.time() - end)
            if isinstance(output, tuple):
                output = output[0]
            output = output.to(torch.float32)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=-1))
    latency = batch_time.avg / batch_size * 1000
    perf = batch_size / batch_time.avg
    print('Latency: %.3f ms'%latency)
    print("Throughput: {:.3f} fps".format(perf))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TransNetV2 inference')
    parser.add_argument('--dataset', type=str, default='RAI', help='dataset name')
    parser.add_argument('--data_path', '-d', type=str, default=None,
                        help='path to dataset (default: None)')
    parser.add_argument("--train_dir", default="train", type=str, help="name of train dir")
    parser.add_argument("--val_dir", default="", type=str, help="name of val dir")
    parser.add_argument("--cache_dataset", dest="cache_dataset", default=False, action="store_true",
                        help='Cache the datasets for quicker initialization. '
                            'It also serializes the transforms')

    parser.add_argument('--checkpoint', type=str, default='transnetv2-pytorch-weights.pth',
                        help='path to model checkpoint file, default is None')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch_size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('-p', '--print_freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--thr', default=0.5, type=float, help='threshold for transition')

    parser.add_argument('--ipex', action='store_true', default=False,
                        help='use intel pytorch extension')
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
    args = parser.parse_args()

    dataloader = None
    model = TransNetV2()
    model.eval()

    inference(model, dataloader, args)
