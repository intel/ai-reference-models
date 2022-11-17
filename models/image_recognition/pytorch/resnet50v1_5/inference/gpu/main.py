#
# ****************************************************************************
# Copyright 2019-2022 Intel Corporation
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
import sys
import random
import shutil
import time
import warnings
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import Subset
import math

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

cwd = os.path.dirname(os.path.abspath(__file__))
hub = os.path.expanduser("~/.cache/torch/intel")
if not os.path.exists(hub):
    os.makedirs(hub)

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', nargs='?', default='imagenet',
                    help='path to dataset (default: imagenet)')
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
parser.add_argument('-p', '--print-freq', default=1, type=int,
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
parser.add_argument('--dist-url', default='env://', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='ccl', type=str,
                    help='distributed backend, default is torch-ccl')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--xpu', default=None, type=int,
                    help='XPU id to use.')
parser.add_argument('--tf32', default=0, type=int, help='Datatype used: TF32')
parser.add_argument('--bf32', default=0, type=int, help='Datatype used: BF32')
parser.add_argument('--fp16', default=0, type=int, help='Datatype used: FP16')
parser.add_argument('--bf16', default=0, type=int, help='Datatype used: BF16')
parser.add_argument('--int8', default=0, type=int, help='Use int8 quantization to do inference')
parser.add_argument('--jit-cache', type=str, default=str(hub), help="path to save/load jit model")
parser.add_argument('--jit-trace', action='store_true',
                    help='enable PyTorch jit trace graph mode')
parser.add_argument('--calib-iters', default=8, type=int,
                    help='iteration number for calibration')
parser.add_argument('--calib-bs', default=32, type=int,
                    metavar='N', help='mini-batch size for calibration')
parser.add_argument('--perchannel-weight', default=False,
                    help='do calibration with weight per channel quantization')
parser.add_argument('--channels-last', action='store_true', help='enable channels last')
parser.add_argument('--num-iterations', default=0, type=int)
parser.add_argument('--tensorboard', default=None, action='store_true',
                    help='Use Tensorboard to visualize the training metrics')
parser.add_argument("--dummy", action="store_true", help='use dummy data for '
                    'benchmark training or val')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--benchmark', default=0, type=int, help='for int8 benchmark '
                    'performance, move H2D out of E2E time')

best_acc1 = 0

def main():
    args = parser.parse_args()

    if args.xpu is not None and args.gpu is not None:
        print('You need to choose running on NV GPU or XPU.')
        sys.exit()

    if args.gpu is not None and not torch.cuda.is_available():
        print('Make sure cuda is enabled in torch.')
        sys.exit()

    if args.xpu is not None:
        import intel_extension_for_pytorch as ipex

    # only for training
    if not args.evaluate:
        if args.tf32:
            print('doing TF32 training')
            torch.xpu.set_fp32_math_mode(torch.xpu.FP32MathMode.TF32)
        elif args.bf32:
            args.bf16 = 1
            print('doing BF32 training')
            torch.xpu.set_fp32_math_mode(torch.xpu.FP32MathMode.BF32)
        else:
            torch.xpu.set_fp32_math_mode(torch.xpu.FP32MathMode.FP32)

    if args.dist_backend == 'ccl':
        try:
            import oneccl_bindings_for_pytorch
        except ImportError:
            print("oneccl_bindings_for_pytorch not available!")

    if args.int8 and (not args.evaluate or args.xpu is None):
        print('For int8 quantization, it is only used in XPU inference, '
              'you need to pass -e and --xpu [dev_id] in your command')
        sys.exit()

    if args.int8 and args.channels_last:
        print('For int8 quantization, channels last is not supported for now')
        sys.exit()

    if args.tensorboard is not None:
        from torch.utils.tensorboard import SummaryWriter
        global writer
        writer = SummaryWriter(log_dir='./tensorboard_log')
        if args.num_iterations is not None:
            warnings.warn('Tensorboard is displaying at epoch unit.')

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        mpi_world_size = int(os.environ.get('PMI_SIZE', -1))

        if mpi_world_size > 0:
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['RANK'] = os.environ.get('PMI_RANK', -1)
            os.environ['WORLD_SIZE'] = os.environ.get('PMI_SIZE', -1)
            args.rank = int(os.environ.get('PMI_RANK', -1))
        args.world_size = int(os.environ.get("WORLD_SIZE", -1))

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    # 1 XPU card has 2 tile, and both are regarded as isolated devices/nodes
    ngpus_per_node = 1
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(ngpus_per_node, args)

def jit_calib(model, val_loader_calib, args):
    print('doing int8 jit calibration')
    jit_model_file = os.path.join(args.jit_cache, "rn50_jit_model_int8.pt")
    if os.path.isfile(jit_model_file):
        print("=> load jit model from {}".format(jit_model_file))
        modelJit = torch.load(jit_model_file)
        print("=> load jit model ... done")
    else:
        from torch.jit._recursive import wrap_cpp_module
        from torch.quantization.quantize_jit import (
            convert_jit,
            prepare_jit,
        )
        modelJit = torch.jit.script(model)
        modelJit = wrap_cpp_module(torch._C._jit_pass_fold_convbn(modelJit._c))

        with torch.inference_mode():
            if args.perchannel_weight:
                qconfig = torch.quantization.QConfig(
                    activation=torch.quantization.observer.MinMaxObserver.with_args(
                        qscheme=torch.per_tensor_symmetric,
                        reduce_range=False,
                        dtype=torch.quint8
                    ),
                    weight=torch.quantization.default_per_channel_weight_observer
                )
            else:
                qconfig = torch.quantization.QConfig(
                    activation=torch.quantization.observer.MinMaxObserver.with_args(
                        qscheme=torch.per_tensor_symmetric,
                        reduce_range=False,
                        dtype=torch.quint8
                    ),
                    weight=torch.quantization.default_weight_observer
                )
            modelJit = prepare_jit(modelJit, {'': qconfig}, True)

            for i, (input, target) in enumerate(val_loader_calib):
                calib = input.to(args.xpu)
                modelJit(calib)

                if i == args.calib_iters - 1:
                    break
            modelJit = convert_jit(modelJit, True)

    return modelJit

def main_worker(ngpus_per_node, args):
    global best_acc1

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + args.gpu

        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

        if args.gpu is not None:
            args.gpu = args.rank
        elif args.xpu is not None:
            args.xpu = args.rank

    if args.gpu is not None:
        print("Use GPU: {}".format(args.gpu))
        args.gpu = "cuda:{}".format(args.gpu)
    elif args.xpu is not None:
        print("Use XPU: {}".format(args.xpu))
        args.xpu = "xpu:{}".format(args.xpu)
    else:
        print("Use CPU")

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    # channels last
    # TODO: this will be default memory format in future
    if args.channels_last:
        print('model is converted to channels last')
        model = model.to(memory_format=torch.channels_last)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs of the current node.
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        elif args.xpu is not None:
            torch.xpu.set_device(args.xpu)
            model.xpu(args.xpu)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        print('model to cuda')
    elif args.xpu is not None:
        torch.xpu.set_device(args.xpu)
        model = model.xpu(args.xpu)
        print('model to xpu')
    else:
        # do training or inference on CPU
        pass

    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss()
    if args.gpu is not None:
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    elif args.xpu is not None:
        criterion = nn.CrossEntropyLoss().xpu(args.xpu)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)

    # torch.xpu.optimize is only for device xpu and no jit script
    if args.xpu is not None:
        if args.evaluate:
            if not args.int8:
                print('doing torch xpu optimize for inference')
                model.eval()
                dtype = torch.float16 if args.fp16 else torch.float32
                dtype = torch.bfloat16 if args.bf16 else dtype
                # FIXME: For FP32 BS1024 inference, it will get -997 runtime error.
                # So half divide the batch size here. JIRA: https://jira.devtools.intel.com/browse/GSD-1869
                sample_batch_size = int(args.batch_size / 2)
                # avoid batch size to be 0 after half divide
                if sample_batch_size == 0:
                    sample_batch_size = 1
                sample_input = torch.randn((sample_batch_size, 3, 224, 224), device=args.xpu)
                model = torch.xpu.optimize(model=model, dtype=dtype, level="O1",
                                           sample_input=sample_input)
        else:
            model.train()
            print('doing torch xpu optimize for training')
            model, optimizer = torch.xpu.optimize(model=model, optimizer=optimizer, level="O1",
                                                  dtype=torch.bfloat16 if args.bf16 else torch.float32)

    if args.distributed:
        if args.xpu is not None:
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.xpu])

    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None or args.xpu is None:
                checkpoint = torch.load(args.resume)
            elif args.gpu is not None:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            elif args.xpu is not None:
                # Map model to be loaded to specified single gpu.
                loc = 'xpu:{}'.format(args.xpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.gpu is not None:
        cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # TODO: when use dummy dataset, the command shoud pass a dir, it needs revision in future
    if args.dummy:
        print("Dummy data is used!")
        train_dataset = datasets.FakeData(1281167, (3, 224, 224), 1000, transforms.ToTensor())
        val_dataset_size = args.num_iterations * args.batch_size if (args.dummy and args.num_iterations) else 50000
        val_dataset = datasets.FakeData(val_dataset_size, (3, 224, 224), 1000, transforms.ToTensor())
    else:
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    # [watch out] The pin memory is default enabled on CUDA for now in torch.
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, pin_memory_device="xpu", sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, pin_memory_device="xpu", sampler=val_sampler)

    # Profiling
    profiling = os.environ.get("PROFILE", "OFF").upper() in ["1", "Y", "ON", "YES", "TRUE"]

    use_autocast = False
    if args.bf16 or args.fp16:
        print('using autocast')
        use_autocast = True

    if args.evaluate:
        if args.int8:
            # calibration dataloader
            val_loader_calib = torch.utils.data.DataLoader(
                val_dataset, batch_size=args.calib_bs, shuffle=False,
                num_workers=args.workers, pin_memory=True)

            # do calibration and return quant model
            model_calib = jit_calib(model, val_loader_calib, args)

            val_loader_inf = torch.utils.data.DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)

            print('doing int8 inference')
            validate_quantization(val_loader_inf, model_calib, criterion, profiling, args)
        else:
            # epoch pass 0
            validate(val_loader, model, criterion, 0, profiling, use_autocast, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        args.lr = scheduler.get_last_lr()[0]

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, profiling, use_autocast, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, epoch, profiling, use_autocast, args)

        scheduler.step()

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
                'scheduler' : scheduler.state_dict()
            }, is_best)

    if args.tensorboard:
        writer.close()

def train(train_loader, model, criterion, optimizer, epoch, profiling, use_autocast, args):
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

    # record throughput
    throughput = 0.0

    data_start = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - data_start)

        if args.channels_last:
            print('input to channels last')
            images = images.to(memory_format=torch.channels_last)

        start_time = time.time()

        if args.xpu is not None:
            # TODO: later the knieto will be used
            with torch.autograd.profiler_legacy.profile(enabled=profiling, use_xpu=True, record_shapes=False) as prof:
                images = images.to(args.xpu)
                target = target.to(args.xpu)

                with torch.xpu.amp.autocast(enabled=use_autocast, dtype=torch.bfloat16):
                    # compute output
                    output = model(images)
                    loss = criterion(output, target)

                # compute gradient and do SGD step
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                # D2H
                if args.xpu is not None:
                    loss = loss.cpu()
                    output = output.cpu()
                    target = target.cpu()

                # sync for time measurement on XPU
                if args.xpu is not None:
                    torch.xpu.synchronize(args.xpu)

            if profiling:
                profile_name = 'fp32'
                if args.fp16:
                    profile_name = 'fp16'
                elif args.bf16:
                    profile_name = 'bf16'
                if args.distributed:
                    profile_name += '.xpu.' + str(args.rank)
                torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"), './profiling.' + profile_name + '.train.pt')
                torch.save(prof.table(sort_by="id", row_limit=100000), './profiling.' + profile_name + '.train.detailed.pt')
        else:
            activities = None
            prof_sort = None
            if profiling:
                prof_sort = "self_cpu_time_total"
                activities=[torch.profiler.ProfilerActivity.CPU]
                if args.gpu is not None:
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                    prof_sort = "self_cuda_time_total"

            with torch.profiler.profile(activities=activities, record_shapes=False) as prof:
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)
                    target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # D2H, as sync
                if args.gpu is not None:
                    loss = loss.cpu()
                    output = output.cpu()
                    target = target.cpu()

            if profiling:
                torch.save(prof.key_averages().table(sort_by=prof_sort), './profiling.card.' + str(args.xpu) + '.pt')

        # measure elapsed time
        duration_train = time.time() - start_time
        batch_time.update(duration_train)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        if i % args.print_freq == 0:
            progress.display(i + 1)

        # exclude first iteration for calculating througput
        if i >= 1:
            throughput += args.batch_size / duration_train
        data_start = time.time()

        if i == (args.num_iterations - 1) and args.num_iterations >= 2:
            print('Training performance: batch size:%d, throughput:%.2f image/sec'
                  % (args.batch_size, throughput / (args.num_iterations - 1)))
            sys.exit(0)
        elif args.num_iterations == 0 and i == len(train_loader) - 1:
            print('Training performance: batch size:%d, throughput:%.2f image/sec'
                  % (args.batch_size, throughput / (len(train_loader) - 2)))
            sys.exit(0)

    if args.tensorboard:
        draw_tensorboard(epoch, losses.avg, top1.avg, top5.avg, 'train', args)

def validate(val_loader, model, criterion, epoch, profiling, use_autocast, args):

    def run_validate(loader, model, base_progress=0):

        # record throughput
        throughput = 0.0

        with torch.no_grad():
            for i, (images, target) in enumerate(loader):
                i = base_progress + i

                if args.channels_last:
                    images = images.to(memory_format=torch.channels_last)
                    print('images convert to channels last')

                start_time = time.time()

                if args.xpu:
                    with torch.autograd.profiler_legacy.profile(enabled=profiling, use_xpu=True, record_shapes=False) as prof:
                        images = images.to(args.xpu)

                        # compute output
                        output = model(images)

                        # sync for time measurement
                        if args.xpu is not None:
                            torch.xpu.synchronize(args.xpu)

                    if profiling:
                        profile_name = 'fp32'
                        if args.fp16:
                            profile_name = 'fp16'
                        elif args.bf16:
                            profile_name = 'bf16'
                        torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"), './profiling.' + profile_name + '.inf.pt')
                        torch.save(prof.table(sort_by="id", row_limit=100000), './profiling.' + profile_name + '.inf.detailed.pt')
                else:
                    activities = None
                    prof_sort = None
                    if profiling:
                        prof_sort = "self_cpu_time_total"
                        activities=[torch.profiler.ProfilerActivity.CPU]
                        if args.gpu is not None:
                            activities.append(torch.profiler.ProfilerActivity.CUDA)
                            prof_sort = "self_cuda_time_total"

                    with torch.profiler.profile(activities=activities, record_shapes=False) as prof:
                        if args.gpu is not None:
                            images = images.cuda(args.gpu, non_blocking=True)

                        # compute output
                        output = model(images)

                        # sync for time measurement
                        if args.gpu is not None:
                            torch.cuda.synchronize(args.gpu)

                    if profiling:
                        torch.save(prof.key_averages().table(sort_by=prof_sort), './profiling.pt')

                # D2H
                output = output.cpu()

                # measure elapsed time
                duration_eval = time.time() - start_time
                batch_time.update(duration_eval)

                loss = criterion(output.float(), target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output.float(), target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                if i % args.print_freq == 0:
                    progress.display(i + 1)

                # exclude first iteration for calculating througput
                if i >= 1:
                    throughput += args.batch_size / duration_eval

                if i == (args.num_iterations - 1) and args.num_iterations >= 2:
                    print('Evalution performance: batch size:%d, throughput:%.2f image/sec, Acc@1:%.2f, Acc@5:%.2f'
                        % (args.batch_size, throughput / (args.num_iterations - 1), top1.avg, top5.avg))
                    sys.exit(0)
                elif args.num_iterations == 0 and i == len(val_loader) - 1:
                    print('Evalution performance: batch size:%d, throughput:%.2f image/sec, Acc@1:%.2f, Acc@5:%.2f'
                        % (args.batch_size, throughput / (len(val_loader) - 2), top1.avg, top5.avg))
                    sys.exit(0)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    if args.jit_trace:
        autocast_dtype = torch.float32
        if args.fp16:
            autocast_dtype = torch.float16
        elif args.bf16:
            autocast_dtype = torch.bfloat16

        trace_input = torch.randn(args.batch_size, 3, 224, 224).to(args.xpu)
        print('jit trace')
        # TODO: sometimes got -997 issue, JIRA: https://jira.devtools.intel.com/browse/GSD-1869
        with torch.xpu.amp.autocast(enabled=use_autocast, dtype=autocast_dtype, cache_enabled=False):
            model = torch.jit.trace(model, trace_input)

    run_validate(val_loader, model)

    progress.display_summary()

    if args.tensorboard:
        draw_tensorboard(epoch, None, top1.avg, top5.avg, 'val', args)

    return top1.avg

def validate_quantization(val_loader, model, criterion, profiling, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    # record throughput
    throughput = 0.0

    with torch.inference_mode():
        for i, (images, target) in enumerate(val_loader):
            if args.xpu is not None and args.benchmark == 1:
                images = images.to(args.xpu)

            start = time.time()
            with torch.autograd.profiler_legacy.profile(enabled=profiling, use_xpu=True, record_shapes=False) as prof:

                if args.xpu is not None and args.benchmark == 0:
                    images = images.to(args.xpu)

                # compute output
                output = model(images)

                # D2H
                output = output.to("cpu")

                # sync for time measurement
                torch.xpu.synchronize(args.xpu)

                # measure elapsed time
                end = time.time()
                batch_time.update(end - start)
                duration_eval = end - start

            if profiling:
                torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"), './profiling.int8.inf.pt')
                torch.save(prof.table(sort_by="id", row_limit=100000), './profiling.detailed.int8.inf.pt')

            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            if i % args.print_freq == 0:
                progress.display(i + 1)

            # exclude first iteration for calculating througput
            perf_start_iter = 1
            if args.benchmark == 1 and args.num_iterations >= 500:
                perf_start_iter = math.floor(args.num_iterations * 0.7)
            if i >= perf_start_iter:
                throughput += args.batch_size / duration_eval

            if i == (args.num_iterations - 1) and args.num_iterations >= 2:
                print('Quantization Evalution performance: batch size:%d, throughput:%.2f image/sec, Acc@1:%.2f, Acc@5:%.2f'
                    % (args.batch_size, throughput / (args.num_iterations - perf_start_iter), top1.avg, top5.avg))
                sys.exit(0)
            elif args.num_iterations == 0 and i == len(val_loader) - 1:
                print('Quantization Evalution performance: batch size:%d, throughput:%.2f image/sec, Acc@1:%.2f, Acc@5:%.2f'
                    % (args.batch_size, throughput / (len(val_loader) - 2), top1.avg, top5.avg))
                sys.exit(0)

        progress.display_summary()

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def qnormalize(tensor, mean, std, scl):
    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if mean.ndim == 1:
        mean = mean[:, None, None]
    if std.ndim == 1:
        std = std[:, None, None]

    tensor.sub_(mean).div_(std)#.mul_(scale)#tensor.sub_(255 * mean).mul(128/255*(1/(1-0.406)))
    out = torch.quantize_per_tensor(tensor, scale=scl, zero_point=0, dtype=torch.qint8)
    return out

def compute_scale(val_loader_com):
    for i, (input, target) in enumerate(val_loader_com):
        scale = 1.0 / (128 / torch.max(input))
        return scale

class QNormalize(object):
    def __init__(self, mean, std, scale):
        self.mean = mean
        self.std = std
        self.scale = scale

    def __call__(self, tensor):
        return qnormalize(tensor, self.mean, self.std, self.scale)

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
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

    def all_reduce(self):
        total = torch.FloatTensor([self.sum, self.count])
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

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

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        pred = pred.cpu()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def draw_tensorboard(num_epoch, avg_loss, avg_acc1, avg_acc5, mode, args):
    if mode == 'train':
        writer.add_scalar('training: learning rate', args.lr, num_epoch)
        writer.add_scalar('training: loss', avg_loss, num_epoch)
        writer.add_scalar('training: top1 acc', avg_acc1, num_epoch)
        writer.add_scalar('training: top5 acc', avg_acc5, num_epoch)
    else:
        writer.add_scalar('val: top1 acc', avg_acc1, num_epoch)
        writer.add_scalar('val: top5 acc', avg_acc5, num_epoch)

if __name__ == '__main__':
    main()
