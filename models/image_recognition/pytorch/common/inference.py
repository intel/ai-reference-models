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
import torchvision.models as models
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
from dataset import get_imagenet_test_data_loader

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

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

def inference(model, dataloader, datatype, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    batch_size = args.batch_size
    warmup_iters = args.warmup_iterations
    max_iters = len(dataloader) if args.max_iterations is None else args.max_iterations
    progress = ProgressMeter(
        max_iters,
        [batch_time, losses, top1, top5],
        prefix='Test: ')
    model.eval()
    if args.ipex:
        import intel_extension_for_pytorch as ipex
        model = model.to(memory_format=torch.channels_last)
        model = ipex.optimize(model, dtype=datatype, level="O1")
    else:
        if args.jit:
            model = model.to(memory_format=torch.channels_last)
        else:
            from torch.utils import mkldnn as mkldnn_utils  
            model = mkldnn_utils.to_mkldnn(model, dtype=datatype)
    if args.jit:
        if dataloader is None:
            x = torch.randn(batch_size, 3, args.height, args.width)
        else:
            for i, batch in enumerate(dataloader):
                x = torch.randn(batch[0].shape)
                break
        x = x.to(memory_format=torch.channels_last)
        if args.precision == "bf16":
            with torch.cpu.amp.autocast(), torch.no_grad():
                model = torch.jit.trace(model, x, strict=False)
            model = torch.jit.freeze(model)
        else:
            with torch.no_grad():
                model = torch.jit.trace(model, x, strict=False)
            model = torch.jit.freeze(model)
    with torch.no_grad():
        if dataloader is None:
            for i in range(max_iters):
                images = torch.randn(batch_size, 3, args.height, args.width)
                if i > warmup_iters:
                    end = time.time()
                if not args.ipex and not args.jit:
                    images = images.to(datatype)
                else:
                    images = images.to(memory_format=torch.channels_last)
                if args.ipex and args.precision == "bf16" and not args.jit:
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
                    batch_time.update(time.time() - end)
                if i % args.print_freq == 0:
                    progress.display(i)
        else:
            # warm up
            for i, (images, target) in enumerate(dataloader):
                if i > warmup_iters:
                    break
                if not args.ipex and not args.jit:
                    images = images.to(datatype).to(memory_format=torch.channels_last)
                if args.ipex and args.precision == "bf16" and not args.jit:
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

            criterion = nn.CrossEntropyLoss()
            for i, (images, target) in enumerate(dataloader):
                end = time.time()
                if not args.ipex and not args.jit:
                    images = images.to(datatype).to(memory_format=torch.channels_last)
                if args.ipex and args.precision == "bf16" and not args.jit:
                    output = model(images)
                else:
                    output = model(images)
                batch_time.update(time.time() - end)
                if args.precision == "bf16":
                    output = output.to(torch.float32)
                loss = criterion(output, target)
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
                if max_iters != -1 and i >= max_iters:
                    break
                if i % args.print_freq == 0:
                    progress.display(i)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=-1))
    latency = batch_time.avg / batch_size * 1000
    perf = batch_size / batch_time.avg
    print('Latency: %.3f ms'%latency)
    print("Throughput: {:.3f} fps".format(perf))
    print("Accuracy: {top1.avg:.3f} ".format(top1=top1))

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch models inference')
    parser.add_argument('--data_path', '-d', type=str, default=None,
                        help='path to dataset (default: None)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet18)')
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
    parser.add_argument('-m', '--max_iterations', default=None, type=int, 
                        help='number of max iterations to run')
    parser.add_argument('--log-path', required = False, default = "", type=str,
                        help="Path for the log file")
    parser.add_argument("--cache-dataset", dest="cache_dataset", default=False, action="store_true",
                        help='Cache the datasets for quicker initialization. ' 
                            'It also serializes the transforms')
    args = parser.parse_args()
    model = models.__dict__[args.arch](pretrained=args.pretrained)
    datatype=torch.float32
    if args.precision == 'bf16':
        datatype=torch.bfloat16
    dataloader = get_imagenet_test_data_loader(args)
    inference(model, dataloader, datatype, args)
