#
# ****************************************************************************
# Copyright 2019-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ****************************************************************************

# ****************************************************************************
# BSD 3-Clause License
# # This model is originally from the PyTorch Examples repo
# (https://github.com/pytorch/examples/blob/master/imagenet/main.py)
# ****************************************************************************

# ****************************************************************************
# Copyright (c) 2017,
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ****************************************************************************

import argparse
import os
import random
import shutil
import time
import warnings
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', '-d', type=str, default=None,
                    help='path to dataset (default: None)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--ipex', action='store_true', default=False,
                    help='use intel pytorch extension')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disable CUDA')
parser.add_argument('--precision', default='fp32', type=str,
                    help='set precision of running model, can be fp32, bf16 or int8(todo)')
parser.add_argument('--jit', action='store_true', default=False,
                    help='enable jit fusion path')
parser.add_argument("--dummy", action='store_true',
                    help="using  dummu data to test the performance of inference")
parser.add_argument('-w', '--warmup-iterations', default=30, type=int, metavar='N',
                    help='number of warmup iterations to run')
parser.add_argument('--log-path', required = False, default = "", type=str,
                    help="Path for the log file")
best_acc1 = 0
log_fl = None
def main():
    global log_fl
    log_fl = sys.stdout
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args)
    if args.ipex:
        print("enabling intel pytorch extension path will get good performance")
        import intel_pytorch_extension
        import intel_pytorch_extension as ipex
    if args.precision == 'bf16':
        assert args.ipex, 'please first enable ipex by add option --ipex to run bfloat16 path'
        print("enabling intel pytorch extension mix precision(fp32+bf16)path")
        ipex.enable_auto_mixed_precision(mixed_dtype = torch.bfloat16,
                                      train = False if args.evaluate else True)

    if args.jit:
        assert args.evaluate, 'jit fusion path only support evaluation step'

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None and args.cuda:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    if not args.log_path == "":
        if not os.path.isdir(os.path.dirname(args.log_path)):
            os.makedirs(os.path.dirname(args.log_path))
        log_fl = open(args.log_path, "w")
    ngpus_per_node = torch.cuda.device_count() if args.cuda else 0
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)
    log_fl.write("Done")
    if log_fl != sys.stdout:
        log_fl.write("Closing")
        log_fl.close()


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    global log_fl

    args.gpu = gpu

    if args.gpu is not None:
        log_fl.write("Use GPU: {} for training\n".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        log_fl.write("=> using pre-trained model '{}'\n".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        log_fl.write("=> creating model '{}'\n".format(args.arch))
        model = models.__dict__[args.arch]()

    if not torch.cuda.is_available():
        log_fl.write('using CPU, this will be slow\n')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None and args.cuda:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            if args.cuda:
                model.cuda()
                log_fl.write("create DistributedDataParallel in GPU\n")
            else:
                log_fl.write("create DistributedDataParallel in CPU\n")
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None and args.cuda:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            if args.cuda:
                model.cuda()
        else:
            model = torch.nn.DataParallel(model)
            if args.cuda():
                model.cuda()

    if args.ipex:
        model = model.to(device = 'xpu')
    # define loss function (criterion) and optimizer

    criterion = nn.CrossEntropyLoss()
    if args.cuda:
        criterion = criterion.cuda(args.gpu)
    elif args.ipex:
        criterion = criterion.to(device = 'xpu')

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            log_fl.write("=> loading checkpoint '{}'\n".format(args.resume))
            if args.gpu is None and args.cuda:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None and args.cuda:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            log_fl.write("=> loaded checkpoint '{}' (epoch {})\n"
                  .format(args.resume, checkpoint['epoch']))
        else:
            log_fl.write("=> no checkpoint found at '{}'\n".format(args.resume))

    if args.cuda:
        cudnn.benchmark = True

    # Data loading code
    if not args.dummy:
        assert args.data != None, "please set dataset path if you want to using real data"

        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        if not args.evaluate:
            traindir = os.path.join(args.data, 'train')

            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))

            if args.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            else:
                train_sampler = None

            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    else:
        assert args.evaluate, "please using real dataset if you want run training step"
        train_loader = None
        val_loader = None

    if args.evaluate:
        if args.ipex:
            log_fl.write("using ipex model to do inference\n")
        if args.jit:
            log_fl.write("running jit fusion path\n")
            script_model = torch.jit.script(model)

        if args.jit:
            validate(val_loader, script_model, criterion, args)
        else:
            validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, args):
    global log_fl

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)
        if args.ipex:
            images = images.to(device = 'xpu')
            target = target.to(device = 'xpu')

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    global log_fl

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    if args.dummy:
        number_iter = 50
    else:
        number_iter = len(val_loader)

    progress = ProgressMeter(
        number_iter,
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    if args.dummy:
        log_fl.write("using dummy input data to run\n")
        images = torch.randn(args.batch_size, 3, 224, 224)
        target = torch.arange(1, args.batch_size + 1).long()
        if args.gpu is not None and args.cuda:
            images = images.cuda(args.gpu, non_blocking=True)
        if args.cuda:
            arget = target.cuda(args.gpu, non_blocking=True)
        if args.ipex:
            images = images.to(device = 'xpu')
            target = target.to(device = 'xpu')

        with torch.no_grad():
            for i in range(number_iter):
                if i >= args.warmup_iterations:
                    end = time.time()
                # compute output
                output = model(images)
                if i >= args.warmup_iterations:
                    batch_time.update(time.time() - end)

                #log_fl.write(output)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                if i % args.print_freq == 0:
                    progress.display(i)
    else:
        log_fl.write("using real dataset to run\n")
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(val_loader):
                if args.gpu is not None and args.cuda:
                    images = images.cuda(args.gpu, non_blocking=True)
                if args.cuda:
                    target = target.cuda(args.gpu, non_blocking=True)
                if args.ipex:
                    images = images.to(device = 'xpu')
                    target = target.to(device = 'xpu')
                # compute output
                output = model(images)
                #log_fl.write(output)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i)

    batch_size = args.batch_size
    latency = batch_time.avg / batch_size * 1000
    perf = batch_size / batch_time.avg
    log_fl.write('inference latency %.3f ms\n'%latency)
    log_fl.write('inference performance %.3f fps\n'%perf)

    # TODO: this should also be done with the ProgressMeter
    log_fl.write(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}\n'
          .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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
        global log_fl

        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        log_fl.write('\t'.join(entries) + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
