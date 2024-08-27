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
import torch.distributed as dist
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import math

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# for convergence
converged = False
final_top1_acc = 0.0
final_top5_acc = 0.0

cwd = os.path.dirname(os.path.abspath(__file__))
hub = os.path.expanduser("~/.cache/torch/intel")
if not os.path.exists(hub):
    os.makedirs(hub, exist_ok=True)

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
parser.add_argument('--eval-start-epoch', default=0, type=int, metavar='N',
                    help='epoch start to run validation')
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
parser.add_argument('--dist-url', default='127.0.0.1', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-port', default='29500', type=str,
                    help='url port used to set up distributed training')
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
parser.add_argument('--optimize', action='store_true', help='Use torch.xpu.optimize for training/inference xpu model')
parser.add_argument("--kineto_profile", action="store_true", help="Whether to running kineto profiler",)
parser.add_argument('--disable-broadcast-buffers', action='store_true', help='disable syncing buffers')
parser.add_argument('--bucket-cap', default=25, type=int, help='controls the bucket size in MegaBytes')
parser.add_argument('--large-first-bucket', action="store_true", help='Configure a large capacity of the first bucket in DDP for allreduce')
parser.add_argument("--use-gradient-as-bucket-view", action='store_true', help="Turn ON gradient_as_bucket_view optimization in DDP")
parser.add_argument('--jit-cache', type=str, default=str(hub), help="path to save/load jit model")
parser.add_argument('--jit-trace', action='store_true',
                    help='enable PyTorch jit trace graph mode')
parser.add_argument('--dynamo', action='store_true', help='Use torch.compile to optimize model, inductor backend default')
parser.add_argument('--calib-iters', default=8, type=int,
                    help='iteration number for calibration')
parser.add_argument('--calib-bs', default=32, type=int,
                    metavar='N', help='mini-batch size for calibration')
parser.add_argument('--perchannel-weight', default=False,
                    help='do calibration with weight per channel quantization')
parser.add_argument('--non-blocking', default=False, action='store_true',
                    help='non blocking H2D for input and target, now for int8, default False')
parser.add_argument('--channels-last', action='store_true', help='enable channels last')
parser.add_argument('--num-iterations', default=0, type=int)
parser.add_argument('--converge', default=None, action='store_true',
                    help='Use Tensorboard to visualize the training metrics')
parser.add_argument('--step-size', default=30, type=int, help='LR decay step size')
parser.add_argument('--step-gamma', default=0.1, type=float, help='set the step gamma')
parser.add_argument('--last-step-boundary', default=80, type=int, help='last epoch to decay the LR')
parser.add_argument('--warm-up-epoch', default=0, type=int, help='warm up epochs number for convergence')
parser.add_argument('--decay-epochs', default=33, type=int, metavar='N',
                    help='number of decay epochs to run for lars')
parser.add_argument('--lars', default=False, action='store_true', help='use lars for training')
parser.add_argument('--lars-eta', default=0.0, type=float, help='set the lars epsilon')
parser.add_argument('--skip-checkpoint', default=False, action='store_true', help='skip checkpoint saving')
parser.add_argument('--skip-tensorboard', default=False, action='store_true', help='skip tensorboard')
parser.add_argument('--label-smoothing', default=0.0, type=float)
parser.add_argument('--dummy', action="store_true", help='use dummy data for '
                    'benchmark training or val')
parser.add_argument('--lr-scheduler', default='step', type=str,
                    help='choose lr scheduler, default step, can choose pow')
parser.add_argument('--power-factor', default=1.0, type=float,
                    help='power factor for lr decay policy')
parser.add_argument('--eval-period', default=1, type=int, help='period for doing online evaluation')
parser.add_argument('--eval-offset', default=0, type=int, help='offset for doing online evaluation')
parser.add_argument('--sota-target', default=75.9, type=float, help='set the lars epsilon')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--benchmark', default=0, type=int, help='for int8 benchmark '
                    'performance, move H2D out of E2E time')
parser.add_argument("--save", help='Path to save entile model, save infernce mode, training is not available')
parser.add_argument("--load", help='Path to load entile inference model')
parser.add_argument("--asymm", action='store_true', help='Asymmetric quantization')
parser.add_argument("--uint8", action='store_true', help="Use u8 quantization")
parser.add_argument('--end-lr', type=float, default=1e-4,
                    help='the end learning rate')

# used for record best accrucy after validation
best_acc1 = 0.0
tensorboard_data = {'epoch': 0,
                    'train': {'loss': 0.0, 'top1': 0.0, 'top5': 0.0},
                    'eval': {'loss': 0.0, 'top1': 0.0, 'top5': 0.0}}
global_lr = 0.0
global_num_iter = 0

def main():
    args = parser.parse_args()

    # print all info for running convergence
    if args.converge:
        print('[info] ------------------ converge arguments ------------------')
        print('running model:  ', args.arch)
        print('workers:        ', args.workers)
        print('running bf16:   ', args.bf16)
        print('total epochs:   ', args.epochs)
        print('warm up epoch:  ', args.warm_up_epoch)
        print('eval epoch:     ', args.eval_start_epoch)
        print('batch size:     ', args.batch_size)
        print('initial lr:     ', args.lr)
        print('lr scheduler:   ', args.lr_scheduler)
        if args.lr_scheduler == 'step':
            print('lr step size:   ', args.step_size)
            print('lr step gamma:  ', args.step_gamma)
            print('lr step boundary:', args.last_step_boundary)
        elif args.lr_scheduler == 'pow':
            print('lr pow factor:  ', args.power_factor)
        else:
            pass
        if args.lars:
            print('using lars:     ', 'True')
            print('choose lars eta:', args.lars_eta)
            print("decay epochs:", args.decay_epochs)
        else:
            print('using sgd:      ', 'True')
        print('label smoothing:', args.label_smoothing)
        print('momentum:       ', args.momentum)
        print('weight decay:   ', args.weight_decay)
        print('seed:           ', args.seed)
        print('eval period:    ', args.eval_period)
        print('eval offset:    ', args.eval_offset)
        print('sota target:    ', args.sota_target)
        print('skip ckpt:      ', args.skip_checkpoint)
        print('disable broadcast: ', args.disable_broadcast_buffers)
        print('large 1st bucket: ', args.large_first_bucket)
        print('use grad as bucket view: ', args.use_gradient_as_bucket_view)
        print('[info] --------------------------------------------------------')

    if args.xpu is not None:
        import intel_extension_for_pytorch as ipex

    # only for xpu training to set the math mode
    if args.xpu is not None:
        if not args.evaluate:
            if args.tf32:
                torch.xpu.set_fp32_math_mode(torch.xpu.FP32MathMode.TF32)
            elif args.bf32:
                args.bf16 = 1
                torch.xpu.set_fp32_math_mode(torch.xpu.FP32MathMode.BF32)
            else:
                torch.xpu.set_fp32_math_mode(torch.xpu.FP32MathMode.FP32)

        if args.dist_backend == 'ccl':
            try:
                import oneccl_bindings_for_pytorch
            except ImportError:
                pass

    if args.converge is not None:
        from torch.utils.tensorboard import SummaryWriter
        global writer
        writer = SummaryWriter(log_dir='./tensorboard_log')
        if args.num_iterations is not None:
            warnings.warn('Tensorboard is displaying at epoch unit.')

    if args.seed is not None:
        print('[info] setting the seed: ', args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    if args.world_size == -1:
        mpi_world_size = int(os.environ.get('PMI_SIZE', -1))

        if mpi_world_size > 0:
            os.environ['MASTER_ADDR'] = args.dist_url #'127.0.0.1'
            os.environ['MASTER_PORT'] = args.dist_port #'29500'
            os.environ['RANK'] = os.environ.get('PMI_RANK', -1)
            os.environ['WORLD_SIZE'] = os.environ.get('PMI_SIZE', -1)
            args.rank = int(os.environ.get('PMI_RANK', -1))
        args.world_size = int(os.environ.get("WORLD_SIZE", -1))
    else: # mpich set
        if 'PMIX_RANK' in os.environ.keys(): # mpich set
            os.environ['MASTER_ADDR'] = args.dist_url #'127.0.0.1'
            os.environ['MASTER_PORT'] = args.dist_port #'29500'
            os.environ['RANK'] = os.environ.get('PMIX_RANK')
            os.environ['WORLD_SIZE'] = str(args.world_size)
            args.rank = int(os.environ.get('PMIX_RANK', -1))

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

        qscheme = torch.per_tensor_affine if args.asymm else torch.per_tensor_symmetric
        dtype = torch.quint8 if args.uint8 else torch.qint8
        with torch.inference_mode():
            if args.perchannel_weight:
                qconfig = torch.quantization.QConfig(
                    activation=torch.quantization.observer.MinMaxObserver.with_args(
                        qscheme=qscheme,
                        reduce_range=False,
                        dtype=dtype
                    ),
                    weight=torch.quantization.default_per_channel_weight_observer
                )
            else:
                qconfig = torch.quantization.QConfig(
                    activation=torch.quantization.observer.MinMaxObserver.with_args(
                        qscheme=qscheme,
                        reduce_range=False,
                        dtype=dtype
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
        if args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + args.gpu
        init_method = 'tcp://' + args.dist_url + ':' + args.dist_port
        dist.init_process_group(backend=args.dist_backend, init_method=init_method,
                                world_size=args.world_size, rank=args.rank)

        if args.gpu is not None:
            args.gpu = args.rank
        elif args.xpu is not None:
            if 'MPI_LOCALRANKID' in os.environ.keys():
                local_rank = os.environ['MPI_LOCALRANKID']
            elif 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ.keys():
                local_rank = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
            else: # mpich set
                local_rank = os.environ['PALS_LOCAL_RANKID']
            args.xpu = local_rank
            print('world_size:{}, rank:{}, local_rank:{}'.format(args.world_size, args.rank, local_rank))

    if args.gpu is not None:
        args.gpu = "cuda:{}".format(args.gpu)
    elif args.xpu is not None:
        args.xpu = "xpu:{}".format(args.xpu)

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # create model
    if args.load:
        if os.path.isfile(args.load):
            load_path = args.load
            if args.jit_trace:
                model = torch.jit.load(load_path)
            elif args.evaluate and args.int8:
                model = torch.jit.load(load_path)
            else:
                model = torch.load(load_path)
                optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                            momentum=args.momentum,
                                            weight_decay=args.weight_decay)
                scheduler = StepLR(optimizer, step_size=args.step_size, gamma=0.1)
        else:
            print("=> no saved model found at '{}'".format(args.load))
            sys.exit(1)
    else:
        if args.pretrained:
            print("=> using pre-trained model '{}'".format(args.arch))
            model = models.__dict__[args.arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(args.arch))
            model = models.__dict__[args.arch]()

        # channels last
        if args.channels_last:
            print('[info] model is converted to channels last')
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
            print('[info] running on cuda')
        elif args.xpu is not None:
            torch.xpu.set_device(args.xpu)
            model = model.xpu(args.xpu)
            print('[info] running on xpu')
        else:
            print('[info] running on cpu')

        # define optimizer, and learning rate scheduler
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=args.momentum, weight_decay=args.weight_decay)

        # for xpu, lars can be used for training
        if args.lars and args.xpu is not None:
            eeta = args.lars_eta
            print('[info] using lars, eeta = ', eeta)
            optimizer = torch.xpu.optim.Lars(model.parameters(),
                                             lr=args.lr,
                                             momentum=args.momentum,
                                             weight_decay=args.weight_decay,
                                             eeta=eeta)

        # xpu model optimize solution
        if args.xpu is not None:
            import intel_extension_for_pytorch as ipex
            using_block_layout = os.environ.get("IPEX_XPU_ONEDNN_LAYOUT", "OFF").upper() in ["1", "Y", "ON", "YES", "TRUE"]
            if using_block_layout:
                ipex.disable_auto_channels_last()
                print('[info] using block layout and disable the auto channels last')
            if args.evaluate:
                if not args.int8 and args.optimize:
                    print('[info] doing ipex.optimize for inference')
                    model.eval()
                    dtype = torch.float16 if args.fp16 else torch.float32
                    dtype = torch.bfloat16 if args.bf16 else dtype
                    model = ipex.optimize(model=model, dtype=dtype, level="O1")
            else:
                if args.optimize:
                    model.train()
                    print('[info] doing ipex.optimize for training')
                    model, optimizer = ipex.optimize(model=model, optimizer=optimizer, level="O1",
                                                     dtype=torch.bfloat16 if args.bf16 else torch.float32)

        if args.distributed:
            if args.xpu is not None:
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int(args.workers / ngpus_per_node)

                if args.large_first_bucket:
                    # set the first bucket with maximal size to cover all parameters for allreduce
                    dist._DEFAULT_FIRST_BUCKET_BYTES = sys.maxsize
                model = torch.nn.parallel.DistributedDataParallel(model,
                                                                  device_ids=[args.xpu],
                                                                  broadcast_buffers=False if args.disable_broadcast_buffers else True,
                                                                  bucket_cap_mb=args.bucket_cap,
                                                                  gradient_as_bucket_view=args.use_gradient_as_bucket_view)
            elif args.gpu is not None:
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int(args.workers / ngpus_per_node)

                if args.large_first_bucket:
                    # set the first bucket with maximal size to cover all parameters for allreduce
                    dist._DEFAULT_FIRST_BUCKET_BYTES = sys.maxsize
                model = torch.nn.parallel.DistributedDataParallel(model,
                                                                  device_ids=[args.gpu],
                                                                  broadcast_buffers=False if args.disable_broadcast_buffers else True,
                                                                  bucket_cap_mb=args.bucket_cap,
                                                                  gradient_as_bucket_view=args.use_gradient_as_bucket_view)

        # LR scheduler
        scheduler = StepLR(optimizer=optimizer, step_size=args.step_size, gamma=args.step_gamma)

        # optionally resume from a checkpoint
        if args.resume:
            if os.path.isfile(args.resume):
                print("[info] loading checkpoint '{}'".format(args.resume))
                if args.gpu is not None:
                    # Map model to be loaded to specified single gpu.
                    print('[info] loading checkpoint to ', str(args.gpu))
                    checkpoint = torch.load(args.resume, map_location=args.gpu)
                elif args.xpu is not None:
                    # Map model to be loaded to specified single xpu.
                    print('[info] loading checkpoint to ', str(args.xpu))
                    checkpoint = torch.load(args.resume, map_location=args.xpu)
                else:
                    # cpu load
                    checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint['epoch']
                best_acc1 = checkpoint['best_acc1']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
            else:
                raise RuntimeError("[error] no checkpoint found at '{}'".format(args.resume))

    # create dataset
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if args.dummy:
        print("[info] dummy dataset is used")
        if args.num_iterations > 0:
            train_dataset_size = args.num_iterations * args.batch_size  * (args.world_size if args.distributed else 1)
        else:
            train_dataset_size = 1281167
        train_dataset = datasets.FakeData(train_dataset_size, (3, 224, 224), 1000, transforms.ToTensor())
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

    # create dataloader sampler
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    # create dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=args.workers,
                                               sampler=train_sampler,drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             sampler=val_sampler)
    if args.xpu is not None:
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=(train_sampler is None),
                                                   num_workers=args.workers,
                                                   pin_memory=True,
                                                   pin_memory_device="xpu",
                                                   sampler=train_sampler,drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=args.batch_size,
                                                 shuffle=False,
                                                 num_workers=args.workers,
                                                 pin_memory=True,
                                                 pin_memory_device="xpu",
                                                 sampler=val_sampler)

    # profiling, support both 2 methods to enable profiler
    if args.kineto_profile:
        os.environ["PROFILE"] = "1"
    profiling = os.environ.get("PROFILE", "OFF").upper() in ["1", "Y", "ON", "YES", "TRUE"]

    # autocast
    use_autocast = False
    if args.bf16 or args.fp16:
        use_autocast = True
    autocast_dtype = torch.float32
    if args.fp16:
        autocast_dtype = torch.float16
        print('[info] running autocast: fp16')
    elif args.bf16:
        autocast_dtype = torch.bfloat16
        print('[info] running autocast: bf16')
    else:
        pass

    if args.evaluate:
        print('[info] running inference')
        if args.int8:
            # calibration dataloader
            val_loader_calib = torch.utils.data.DataLoader(
                val_dataset, batch_size=args.calib_bs, shuffle=False,
                num_workers=args.workers, pin_memory=True, pin_memory_device="xpu")

            # do calibration and return quant model
            if args.load:
                model_calib = model
            else:
                model_calib = jit_calib(model, val_loader_calib, args)
            if args.save:
                torch.jit.save(model_calib, args.save)
            validate_quantization(val_loader, model_calib, criterion, profiling, args)
        else:
            validate(val_loader, model, criterion, 0, profiling, use_autocast, autocast_dtype, args)
        return
    else:
        print('[info] running training')

    global_start_time = time.time()

    # warm up for running convergence
    if args.converge and not args.resume and args.warm_up_epoch > 0:
        warm_up_epoch = args.warm_up_epoch
        warm_up_portion = args.lr / float(warm_up_epoch)
        for epoch in range(0, warm_up_epoch):
            if args.lars == False:
                optimizer.param_groups[0]['lr'] = (epoch + 1) * warm_up_portion
            train(train_loader, model, criterion, optimizer, epoch, profiling, use_autocast, autocast_dtype, args, mode='warming')
            print('[info] warmup [', (epoch + 1), '][', warm_up_epoch, '] lr = ', optimizer.param_groups[0]['lr'])

    last_acc = best_acc1
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()

        global global_lr
        global_lr = optimizer.param_groups[0]['lr']

        if args.converge and (not args.distributed or (args.distributed and args.rank == 0)):
            print('[info] Epoch[', epoch, '] start time = ', time.asctime(time.localtime(epoch_start_time)))

        if args.distributed:
            train_sampler.set_epoch(epoch)

        if epoch == args.last_step_boundary and args.lr_scheduler == 'step':
            optimizer.param_groups[0]['lr'] *= args.step_gamma

        if args.converge and (not args.distributed or (args.distributed and args.rank == 0)):
            print('[info] Epoch[', epoch, '] lr = ', optimizer.param_groups[0]['lr'])

        train(train_loader, model, criterion, optimizer, epoch, profiling, use_autocast, autocast_dtype, args, mode='training')

        # evaluate on validation set
        acc1 = last_acc
        if (epoch >= args.eval_start_epoch) and (epoch % args.eval_period == args.eval_offset):
            print('[info] Epoch: ', epoch, ' is doing evaluation')
            acc1 = validate(val_loader, model, criterion, epoch, profiling, use_autocast, autocast_dtype, args)
            last_acc = acc1

        # update the LR
        if args.lars == False:
            scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.skip_checkpoint and \
            (not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0)):
            save_checkpoint(state={
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, is_best=is_best, args=args)

        # show info
        if not args.distributed or (args.distributed and args.rank == 0):
            epoch_end_time = time.time()
            print('[info] Epoch[', epoch, '] end time = ', time.asctime(time.localtime(epoch_end_time)))
            print('[info] Epoch[', epoch, '] consume time = ', ((epoch_end_time - epoch_start_time) / 3600.0), ' hours')

        if converged:
            break

    if not args.distributed or (args.distributed and args.rank == 0):
        global_end_time = time.time()
        print('[info] Global start time = ', time.asctime(time.localtime(global_start_time)))
        print('[info] Global end time = ', time.asctime(time.localtime(global_end_time)))
        print('[info] Global consume time = ', ((global_end_time - global_start_time) / (3600.0)), ' hours')
        if converged:
            print('[Successful] Reach convergence, final top1 acc: ', final_top1_acc)
            print('[Successful] Reach convergence, final top5 acc: ', final_top5_acc)
        else:
            print('[Failed] Miss convergence')

    if args.converge and not args.skip_tensorboard:
        if not args.distributed or (args.distributed and args.rank == 0):
            writer.close()

def train(train_loader, model, criterion, optimizer, epoch, profiling, use_autocast, autocast_dtype, args, mode='training'):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    global global_num_iter

    non_blocking = False
    if args.non_blocking:
        non_blocking = True

    # for xpu + dynamo
    def one_iter(model, images, target):
        if autocast_dtype == torch.bfloat16 or autocast_dtype == torch.float16:
            with torch.autocast(device_type="xpu", enabled=use_autocast, dtype=autocast_dtype):
                # compute output
                output = model(images)
                loss = criterion(output, target)
        else:
            # compute output
            output = model(images)
            loss = criterion(output, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.lars:
            # Update LR for Lars
            MLPerfLRScheduler(optimizer, global_num_iter, len(train_loader), args)
        optimizer.step()
        return [output, loss]

    # set the train mode
    model.train()

    if args.dynamo:
        # TODO: here backend should be 'ipex'
        print('[info] running graph mode')
        train_opt = torch.compile(one_iter, backend="inductor")
    else:
        print('[info] running eager mode')
        train_opt = one_iter

    # record time
    duration_total = 0.0
    warmup_iter = 5

    # config profiler
    import contextlib
    def profiler_setup(profiling=False, *args, **kwargs):
        if profiling:
            return torch.profiler.profile(*args, **kwargs)
        else:
            return contextlib.nullcontext()
    activities = [torch.profiler.ProfilerActivity.CPU]
    if args.xpu is not None:
        activities.append(torch.profiler.ProfilerActivity.XPU)
    elif args.gpu is not None:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    # fixed wait iters = 1, active iters = 1, warmup_iters = 3, at_least 5 iters
    # more iters will be skipped, and repeat will be fixed to 1
    num_iters = args.num_iterations if args.num_iterations else len(train_loader)
    skip_iters = max(num_iters - 5, 0)
    schedule = torch.profiler.schedule(skip_first=skip_iters,
                                       wait=1, warmup=3, active=1)
    def trace_handle(prof):
        profile_name = 'fp32'
        if args.fp16:
            profile_name = 'fp16'
        elif args.bf16:
            profile_name = 'bf16'
        if args.distributed:
            if args.xpu is not None:
                profile_name += '.xpu.' + str(args.rank)
            elif args.gpu is not None:
                profile_name += '.cuda.' + str(args.rank)
            else:
                profile_name += '.cpu.' + str(args.rank)
        if args.xpu is not None:
            torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"), './profiling.' + profile_name + '.train.pt')
        elif args.gpu is not None:
            torch.save(prof.key_averages().table(sort_by="self_cuda_time_total"), './profiling.card.' + str(args.xpu) + '.pt')
        else:
            torch.save(prof.key_averages().table(sort_by="self_cpu_time_total"), './profiling.card.' + str(args.xpu) + '.pt')

    # start profiler, or none while profiling is false
    with profiler_setup(profiling, activities=activities, schedule=schedule, on_trace_ready=trace_handle) as prof:
        data_start = time.time()
        for i, (images, target) in enumerate(train_loader):
            global_num_iter +=1
            # measure data loading time
            data_time.update(time.time() - data_start)

            if args.channels_last:
                print('[info] input to channels last')
                images = images.to(memory_format=torch.channels_last)

            if args.xpu is not None:
                try:
                    import memory_check
                    memory_check.display_mem("xpu:0")
                except:
                    pass
                start_time = time.time()
                images = images.to(args.xpu, non_blocking=non_blocking)
                target = target.to(args.xpu, non_blocking=non_blocking)

                [output, loss] = train_opt(model, images, target)

                loss = loss.cpu()
                output = output.cpu()
                target = target.cpu()
            elif args.gpu is not None:
                start_time = time.time()
                images = images.to(args.gpu, non_blocking=non_blocking)
                target = target.to(args.gpu, non_blocking=non_blocking)

                with torch.cuda.amp.autocast(enabled=use_autocast, dtype=autocast_dtype):
                    output = model(images)
                    loss = criterion(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss = loss.cpu()
                output = output.cpu()
                target = target.cpu()
            else:
                start_time = time.time()
                with torch.cpu.amp.autocast(enabled=use_autocast, dtype=autocast_dtype):
                    output = model(images)
                    loss = criterion(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if profiling:
                prof.step()

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
            if i >= warmup_iter or args.num_iterations == 0 and len(train_loader) <= warmup_iter:
                duration_total += duration_train
            data_start = time.time()

            if i == (args.num_iterations - 1) and args.num_iterations >= warmup_iter:
                print('Training performance: batch size:%d, throughput:%.2f image/sec'
                      % (args.batch_size, (args.batch_size / (duration_total / (args.num_iterations - warmup_iter)))))
                sys.exit(0)
            elif args.num_iterations == 0 and i == len(train_loader) - 1:
                iters = len(train_loader) - warmup_iter if len(train_loader) > warmup_iter else len(train_loader)
                print('Training performance: batch size:%d, throughput:%.2f image/sec'
                      % (args.batch_size, (args.batch_size / (duration_total / iters))))
                if args.converge is None:
                    sys.exit(0)

    if args.converge and not args.skip_tensorboard and mode == 'training':
        global tensorboard_data
        tensorboard_data['epoch'] = epoch
        tensorboard_data['train']['loss'] = losses.avg
        tensorboard_data['train']['top1'] = top1.avg
        tensorboard_data['train']['top5'] = top5.avg

def validate(val_loader, model, criterion, epoch, profiling, use_autocast, autocast_dtype, args):
    from torch._inductor import config

    def compile_model(model, val_loader):
        print("====Before compile model====")
        compiled_model = torch.compile(model, backend="inductor", options={"freezing": True})
        return compiled_model

    def run_validate(loader, model, non_blocking, base_progress=0):

        # record time
        duration_total = 0.0
        warmup_iter = 5
        if (not args.num_iterations == 0) and (args.num_iterations <= warmup_iter):
            raise RuntimeError('At least {} iterations required for performance measure'.format(warmup_iter))

        # config profiler
        import contextlib
        def profiler_setup(profiling=False, *args, **kwargs):
            if profiling:
                return torch.profiler.profile(*args, **kwargs)
            else:
                return contextlib.nullcontext()
        activities = [torch.profiler.ProfilerActivity.CPU]
        if args.xpu is not None:
            activities.append(torch.profiler.ProfilerActivity.XPU)
        elif args.gpu is not None:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        # fixed wait iters = 1, active iters = 1, warmup_iters = 3, at_least 5 iters
        # more iters will be skipped, and repeat will be fixed to 1
        num_iters = args.num_iterations if args.num_iterations else len(loader)
        skip_iters = max(num_iters - 5, 0)
        schedule = torch.profiler.schedule(skip_first=skip_iters,
                                           wait=1, warmup=3, active=1)
        def trace_handle(prof):
            profile_name = 'fp32'
            if args.fp16:
                profile_name = 'fp16'
            elif args.bf16:
                profile_name = 'bf16'
            if args.distributed:
                profile_name += '.xpu.' + str(args.rank)
            if args.xpu is not None:
                print(prof.key_averages().table(sort_by="self_xpu_time_total"))
                torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"), './profiling.' + profile_name + '.inf.pt')
            elif args.gpu is not None:
                print(prof.key_averages().table(sort_by="self_cuda_time_total"))
                torch.save(prof.key_averages().table(sort_by="self_cuda_time_total"), './profiling.card.' + str(args.xpu) + '.pt')
            else:
                print(prof.key_averages().table(sort_by="self_cpu_time_total"))
                torch.save(prof.key_averages().table(sort_by="self_cpu_time_total"), './profiling.card.' + str(args.xpu) + '.pt')

        # start profiler, or none while profiling is false
        with profiler_setup(profiling, activities=activities, schedule=schedule, on_trace_ready=trace_handle) as prof, torch.no_grad():
            for i, (images, target) in enumerate(loader):
                i = base_progress + i

                if args.channels_last:
                    images = images.to(memory_format=torch.channels_last)
                    print('[info] images convert to channels last')

                if args.xpu is not None:
                    try:
                        import memory_check
                        memory_check.display_mem("xpu:0")
                    except:
                        pass
                    start_time = time.time()
                    images = images.to(args.xpu, non_blocking=non_blocking)

                    if args.jit_trace:
                        output = model(images)
                    elif args.dynamo:
                        if autocast_dtype == torch.bfloat16 or autocast_dtype == torch.float16:
                            with torch.autocast(device_type="xpu", enabled=use_autocast, dtype=autocast_dtype):
                                output = model(images)
                        else:
                            output = model(images)
                    else:
                        if autocast_dtype == torch.bfloat16 or autocast_dtype == torch.float16:
                            with torch.autocast(device_type="xpu", enabled=use_autocast, dtype=autocast_dtype):
                                output = model(images)
                        else:
                            output = model(images)

                    torch.xpu.synchronize(args.xpu)
                elif args.gpu is not None:
                    start_time = time.time()
                    images = images.to(args.gpu, non_blocking=non_blocking)
                    with torch.cuda.amp.autocast(enabled=use_autocast, dtype=autocast_dtype):
                        output = model(images)

                    torch.cuda.synchronize(args.gpu)
                else:
                    start_time = time.time()
                    with torch.cpu.amp.autocast(enabled=use_autocast, dtype=autocast_dtype):
                        output = model(images)

                if profiling:
                    prof.step()

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
                if i >= warmup_iter and not (args.num_iterations == 0 and i == len(val_loader) - 1):
                    duration_total += duration_eval

                if i == (args.num_iterations - 1) and args.num_iterations >= warmup_iter:
                    print('Evalution performance: batch size:%d, throughput:%.2f image/sec, Acc@1:%.2f, Acc@5:%.2f'
                        % (args.batch_size, (args.batch_size / (duration_total / (args.num_iterations - warmup_iter))), top1.avg, top5.avg))
                    sys.exit(0)
                elif args.num_iterations == 0 and i == len(val_loader) - 1:
                    if args.converge and args.distributed:
                        top1.all_reduce()
                        top5.all_reduce()
                    print('Evalution performance: batch size:%d, throughput:%.2f image/sec, Acc@1:%.2f, Acc@5:%.2f'
                        % (args.batch_size, (args.batch_size / (duration_total / (len(val_loader) - warmup_iter))), top1.avg, top5.avg))
                    if args.converge:
                        global final_top1_acc
                        global final_top5_acc
                        global converged
                        final_top1_acc = top1.avg
                        final_top5_acc = top5.avg
                        if final_top1_acc >= args.sota_target:
                            converged = True
                    else:
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

    if args.xpu is not None:
        # TODO: jit will be deprecated
        if args.jit_trace and not args.load:
            trace_input = torch.randn(args.batch_size, 3, 224, 224).to(args.xpu)
            print('[info] running jit mode')
            if autocast_dtype == torch.bfloat16 or autocast_dtype == torch.float16:
                with torch.autocast(device_type="xpu", enabled=use_autocast, dtype=autocast_dtype, cache_enabled=False):
                    model = torch.jit.trace(model, trace_input)
            else:
                model = torch.jit.trace(model, trace_input)
        elif args.dynamo:
            print('[info] running graph mode')
            model = compile_model(model, val_loader)
        else:
            print('[info] running eager mode')

    if args.save:
        if args.jit_trace:
            torch.jit.save(model, args.save)
        else:
            torch.save(model, args.save)

    non_blocking = False
    if args.non_blocking:
        non_blocking = True

    run_validate(val_loader, model, non_blocking)

    progress.display_summary()

    if args.converge and not args.skip_tensorboard:
        global tensorboard_data
        tensorboard_data['eval']['loss'] = losses.avg
        tensorboard_data['eval']['top1'] = final_top1_acc
        tensorboard_data['eval']['top5'] = final_top5_acc
        draw_tensorboard(args)

    if args.distributed:
        return final_top1_acc
    else:
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

    # record time
    duration_total = 0.0

    non_blocking = False
    if args.non_blocking:
        non_blocking = True

    # config profiler
    import contextlib
    def profiler_setup(profiling=False, *args, **kwargs):
        if profiling:
            return torch.profiler.profile(*args, **kwargs)
        else:
            return contextlib.nullcontext()
    activities = [torch.profiler.ProfilerActivity.CPU]
    if args.xpu is not None:
        activities.append(torch.profiler.ProfilerActivity.XPU)
    elif args.gpu is not None:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    # fixed wait iters = 1, active iters = 1, warmup_iters = 3, at_least 5 iters
    # more iters will be skipped, and repeat will be fixed to 1
    num_iters = args.num_iterations if args.num_iterations else len(val_loader)
    skip_iters = max(num_iters - 5, 0)
    schedule = torch.profiler.schedule(skip_first=skip_iters,
                                       wait=1, warmup=3, active=1)
    def trace_handle(prof):
        if args.xpu is not None:
            torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"), './profiling.int8.inf.pt')

    # start profiler, or none while profiling is false
    with profiler_setup(profiling, activities=activities, schedule=schedule, on_trace_ready=trace_handle) as prof, torch.inference_mode():
        for i, (images, target) in enumerate(val_loader):
            if args.xpu is not None and args.benchmark == 1:
                images = images.to(args.xpu, non_blocking=non_blocking)

            try:
                import memory_check
                memory_check.display_mem("xpu:0")
            except:
                pass

            start = time.time()

            if args.xpu is not None and args.benchmark == 0:
                images = images.to(args.xpu, non_blocking=non_blocking)

            if args.channels_last:
                images = images.to(memory_format=torch.channels_last)

            # compute output
            output = model(images)

            if args.xpu is not None:
                # D2H
                output = output.to("cpu")

                # sync for time measurement
                torch.xpu.synchronize(args.xpu)

            # measure elapsed time
            end = time.time()
            batch_time.update(end - start)
            duration_eval = end - start

            loss = criterion(output, target)

            if profiling:
                prof.step()

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
                duration_total += duration_eval

            if i == (args.num_iterations - 1) and args.num_iterations >= 2:
                print('Quantization Evalution performance: batch size:%d, throughput:%.2f image/sec, Acc@1:%.2f, Acc@5:%.2f'
                    % (args.batch_size, (args.batch_size / (duration_total / (args.num_iterations - perf_start_iter))), top1.avg, top5.avg))
                sys.exit(0)
            elif args.num_iterations == 0 and i == len(val_loader) - 1:
                print('Quantization Evalution performance: batch size:%d, throughput:%.2f image/sec, Acc@1:%.2f, Acc@5:%.2f'
                    % (args.batch_size, (args.batch_size / (duration_total / (len(val_loader) - 2))), top1.avg, top5.avg))
                sys.exit(0)

        progress.display_summary()

    return top1.avg

def save_checkpoint(state, is_best, args, filename='checkpoint.pth.tar'):
    if args.distributed:
        print('[info] rank ', str(args.rank), ' saving checkpoint')
        filename = 'checkpoint.rank.' + str(args.rank) + '.pth.tar'
    else:
        print('[info] single tile saving checkpoint')
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

def MLPerfLRScheduler(optimizer, step, iteration, args):
    global global_lr
    warmup_iter = args.warm_up_epoch * iteration
    decay_steps = args.decay_epochs * iteration
    power = 2
    if step <= warmup_iter:
       lr_rate = args.lr * (step / warmup_iter)
    else:
       lr_step = min((step - warmup_iter), decay_steps)
       lr_rate = ((args.lr - args.end_lr) * (1-(lr_step/decay_steps)) ** power) + args.end_lr
    global_lr = lr_rate
    optimizer.param_groups[0]['lr'] = global_lr

def draw_tensorboard(args):
    global tensorboard_data
    global global_lr
    if not args.distributed or (args.distributed and args.rank == 0):
        epoch = tensorboard_data['epoch']

        train_loss = tensorboard_data['train']['loss']
        eval_loss = tensorboard_data['eval']['loss']

        train_top1 = tensorboard_data['train']['top1']
        eval_top1 = tensorboard_data['eval']['top1']

        train_top5 = tensorboard_data['train']['top5']
        eval_top5 = tensorboard_data['eval']['top5']

        writer.add_scalars('top1 acc', {'train acc': train_top1, 'eval acc': eval_top1}, epoch)
        writer.add_scalars('top5 acc', {'train acc': train_top5, 'eval acc': eval_top5}, epoch)
        writer.add_scalar('learning rate', global_lr, epoch)
        writer.add_scalars('loss value', {'train loss': train_loss, 'eval loss': eval_loss}, epoch)

if __name__ == '__main__':
    main()
