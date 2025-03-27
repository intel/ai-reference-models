# Copyright (c) 2023-2024 Intel Corporation
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

import argparse
import os
import random
import shutil
import time
import warnings
import threading

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torch.fx.experimental.optimization as optimization
import intel_extension_for_pytorch as ipex

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils import ThroughputBenchmark

from lars import *
from lars_utils import *

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

parser.add_argument("data", metavar="DIR", help="path to dataset")
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet18",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: resnet18)",
)  # can change default to RN50
parser.add_argument(
    "-j",
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
# Distributed training args
parser.add_argument(
    "--world-size",
    default=-1,
    type=int,
    help="number of nodes for distributed training",
)
parser.add_argument(
    "--rank", default=-1, type=int, help="node rank for distributed training"
)
parser.add_argument(
    "--dist-url",
    default="env://",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="ccl", type=str, help="distributed backend"
)
# IPEX args
parser.add_argument(
    "--ipex", action="store_true", default=False, help="use intel pytorch extension"
)
parser.add_argument(
    "--bf16", action="store_true", default=False, help="enable ipex bf16 path"
)
parser.add_argument(
    "--bf32", action="store_true", default=False, help="enable ipex bf32 path"
)
parser.add_argument(
    "--fp16", action="store_true", default=False, help="enable ipex fp16 path"
)
# Learning Hyperparams
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)
parser.add_argument("--base-op", type=str, default="sgd", help="base optimizer name")
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.1,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)  # no need : use --base-lr for SGD too
parser.add_argument(
    "--base-lr", type=float, default=0.0125, help="base learning rate for SGD and LARS"
)
parser.add_argument(
    "--end-lr",
    type=float,
    default=0.0001,
    help="end learning rate for polynomial decay LR schedule",
)
parser.add_argument(
    "--poly-power", type=int, default=2, help="power for polynomial decay LR schedule"
)
parser.add_argument(
    "--epochs", default=90, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--warmup-epochs", type=float, default=5, help="number of warmup epochs"
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--local-batch-size",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the local "
    "batch size of single GPU on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument("--epsilon", type=float, default=0, help="epsilon for optimizer")
parser.add_argument(
    "--bn-bias-separately", action="store_true", default=True, help="skip bn and bias"
)
parser.add_argument(
    "--label-smoothing",
    type=float,
    default=0.1,
    help="label smoothing for cross entropy loss",
)
parser.add_argument(
    "--zero-init-residual",
    action="store_true",
    default=False,
    help="Initialize scale params in BN3 of a residual block to zeros instead ones. "
    "Improves accuracy by 0.2~0.3 percent according to https://arxiv.org/abs/1706.02677"
    "Used by Nvidia, but not part of MLPerf reference ",
)
parser.add_argument(
    "--warmup-iterations",
    default=-1,
    type=int,
    metavar="N",
    help="number of total warmup iterations to run",
)
parser.add_argument(
    "-i",
    "--iterations",
    default=-1,
    type=int,
    metavar="N",
    help="number of total iterations to run",
)
parser.add_argument(
    "--train-no-eval",
    action="store_true",
    default=False,
    help="only train, but not evaluate model on validation set",
)
# Evaluation args
parser.add_argument(
    "--target-acc", default=76, type=float, help="Target validation accuracy"
)

best_acc1 = 0


def main():

    torchvision.set_image_backend("accimage")
    args = parser.parse_args()
    print(args)

    # CCL related
    os.environ["MASTER_ADDR"] = str(os.environ.get("MASTER_ADDR", "127.0.0.1"))
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RANK"] = str(os.environ.get("PMI_RANK", 0))
    os.environ["WORLD_SIZE"] = str(os.environ.get("PMI_SIZE", 1))

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
        print("World size: ", args.world_size)

    args.distributed = args.world_size > 1
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    main_worker(args)


def main_worker(args):

    global best_acc1

    print("=> Creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](zero_init_residual=args.zero_init_residual)

    # for ipex path, always convert model to channels_last for bf16, fp16, bf32, fp32.
    if args.ipex:
        model = model.to(memory_format=torch.channels_last)

    if args.ipex and args.bf32:
        ipex.set_fp32_math_mode(mode=ipex.FP32MathMode.BF32, device="cpu")
        print("using bf32 fmath mode\n")

    # Loss function (criterion)
    # Label smoothing supported in Pytorch >= v1.10.0
    if 0 < args.label_smoothing < 1.0:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer
    if args.base_op.lower() == "lars":
        print("Creating LARS optimizer")
        optimizer = ipex.optim._lars.create_optimizer_lars(
            model=model,
            lr=args.base_lr,
            epsilon=args.epsilon,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            bn_bias_separately=args.bn_bias_separately,
        )
    else:
        print("Creating SGD optimizer")
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

    if args.ipex:
        print("using ipex to do training.....................")
        if args.bf16:
            model, optimizer = ipex.optimize(
                model,
                dtype=torch.bfloat16,
                optimizer=optimizer,
                split_master_weight_for_bf16=False,
            )
        elif args.fp16:
            model, optimizer = ipex.optimize(
                model, dtype=torch.half, optimizer=optimizer
            )
        else:
            model, optimizer = ipex.optimize(
                model, dtype=torch.float32, optimizer=optimizer
            )
    # setup distributed training
    if args.distributed:
        import oneccl_bindings_for_pytorch

        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
        print("Rank and world size: ", dist.get_rank(), " ", dist.get_world_size())
        print("Using local batch size: ", args.local_batch_size)
        print("Using global batch size: ", int(args.local_batch_size * args.world_size))
        print("Create DistributedDataParallel in CPU")
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            find_unused_parameters=True,
            broadcast_buffers=False,
            gradient_as_bucket_view=False,
            bucket_cap_mb=50,
        )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            best_acc1 = checkpoint["best_acc1"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> No checkpoint found at '{}'".format(args.resume))

    # Data loading code
    assert args.data != None, "please set dataset path"
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # Train loader
    traindir = os.path.join(args.data, "train")
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.local_batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    # Validation loader
    valdir = os.path.join(args.data, "val")
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        val_sampler = None

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.local_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_sampler,
    )

    num_steps_per_epoch = len(train_loader)
    if args.base_op.lower() == "lars":
        print("Creating LR scheduler ")
        lr_scheduler = MLPerfLRScheduler(
            optimizer,
            args.epochs,
            args.warmup_epochs,
            num_steps_per_epoch,
            args.base_lr,
            args.end_lr,
            args.poly_power,
        )
    else:
        lr_scheduler = None

    time_to_train = 0

    if args.fp16:
        scaler = torch.cpu.amp.GradScaler()
    else:
        scaler = None

    if args.ipex:
        if args.bf16:
            print("running ipex bfloat16 training step\n")
        elif args.bf32:
            print("running ipex bfloat32 training step\n")
        elif args.fp16:
            print("running ipex float16 training step\n")
        else:
            print("running ipex float32 training step\n")

    total_time = 0
    compute_time = 0

    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            train_sampler.set_epoch(epoch)

        if args.base_op.lower() == "sgd":
            # TODO: add warmup for SGD path
            adjust_learning_rate(optimizer, epoch, args)

        # train for each epoch

        epoch_time = 0

        batch_time = AverageMeter(
            "Time", ":6.3f"
        )  # Track total time = data-load + compute
        data_time = AverageMeter("Data", ":6.3f")  # Track data-load time
        losses = AverageMeter("Loss", ":.4e")
        top1 = AverageMeter("Acc@1", ":6.2f")
        top5 = AverageMeter("Acc@5", ":6.2f")
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch),
        )

        # Train mode
        model.train()

        start = time.time()
        for i, (images, target) in enumerate(train_loader):

            if args.distributed:
                dist.barrier()

            if args.ipex:
                images = images.contiguous(memory_format=torch.channels_last)

            if i % args.print_freq == 0:
                progress.display(i)

            # data loading time
            t1 = time.time() - start
            data_time.update(t1)

            # Forward pass
            if args.ipex and args.bf16:
                with torch.autocast(
                    "cpu",
                ):
                    output = model(images)
                output = output.to(torch.float32)
            elif args.ipex and args.fp16:
                with torch.autocast("cpu", dtype=torch.half):
                    output = model(images)
                output = output.to(torch.float32)
            else:
                output = model(images)
            loss = criterion(output, target)

            # compute gradient and do optimizer step
            optimizer.zero_grad()
            if args.fp16:
                scaler.scale(loss).backward()
                if lr_scheduler:
                    lr_scheduler.step()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if lr_scheduler:
                    lr_scheduler.step()
                optimizer.step()

            # measure elapsed time and reset timer
            t2 = time.time() - start
            batch_time.update(t2)
            epoch_time += t2 - t1
            if (
                args.warmup_iterations < 0
                or (epoch - args.start_epoch) * len(train_loader) + i + 1
                > args.warmup_iterations
            ):
                total_time += t2
                compute_time += t2 - t1

            # measure accuracy and record loss
            acc, counts = accuracy(output, target, topk=(1, 5))
            acc1, acc5 = acc[0], acc[1]
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            start = time.time()

            if (
                args.warmup_iterations > 0
                and args.iterations > 0
                and (epoch - args.start_epoch) * len(train_loader) + i + 1
                >= args.iterations + args.warmup_iterations
            ):
                break
            elif (
                args.warmup_iterations < 0
                and args.iterations > 0
                and (epoch - args.start_epoch) * len(train_loader) + i + 1
                >= args.iterations
            ):
                break

        print("epoch: ", epoch, ", train_time(s): ", epoch_time)
        time_to_train += epoch_time
        print("time_to_train(s): ", time_to_train)

        # evaluate on validation set
        if not args.train_no_eval:
            acc1 = validate(val_loader, model, criterion, epoch, args)

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            if args.rank == 0:
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "arch": args.arch,
                        "state_dict": model.state_dict(),
                        "best_acc1": best_acc1,
                        "optimizer": optimizer.state_dict(),
                    },
                    is_best,
                )

            if best_acc1 > args.target_acc:
                break

        if (
            args.warmup_iterations > 0
            and args.iterations > 0
            and (epoch - args.start_epoch) * len(train_loader) + i + 1
            >= args.iterations + args.warmup_iterations
        ):
            break
        elif (
            args.warmup_iterations < 0
            and args.iterations > 0
            and (epoch - args.start_epoch) * len(train_loader) + i + 1
            >= args.iterations
        ):
            break

    # workaround oneccl bad termination issue.
    if args.distributed:
        dist.destroy_process_group()

    # Compute Throughput
    batch_size = args.local_batch_size
    total_iterations = (
        min((args.epochs - args.start_epoch) * len(train_loader), args.iterations)
        if args.iterations > 0
        else (args.epochs - args.start_epoch) * len(train_loader)
    )
    if args.warmup_iterations > 0 and args.iterations < 0:
        total_iterations -= args.warmup_iterations
    if total_iterations <= 0:
        print(
            "Error! length of Train Data Loader or args.iterations are small than warmup iterations! Please put a larger args.iterations or a smaller args.warmup_iterations."
        )
    perf = batch_size * total_iterations / compute_time
    print(
        f"batch size is {batch_size}, total_iterantions is {total_iterations}, compute_time is {compute_time}"
    )
    print("Training throughput(compute): {:.3f} fps".format(perf))
    perf2 = batch_size * total_iterations / total_time
    print(
        f"batch size is {batch_size}, total_iterantions is {total_iterations}, total_time is {total_time}"
    )
    print("Training throughput(dataload+compute): {:.3f} fps".format(perf2))

    print("final time_to_train(s): ", time_to_train)


"""
Distributed evaluation notes:

Default behavior of DDP is to broadcast named buffers from rank 0 to all ranks 
before FWD pass of each iteration. This ensures that BN stats are identical for all 
ranks at evaluation time, and top1 count can be simply aggregated over all ranks for 
accuracy calculation. TODO: Measure performance impact of broadcast by synchronizing 
BN stats only at validation

Distributed sampler pads the dataset to make the total dataset samples divisible by
the number of ranks and this can impact accuracy calculation. Since the datset is not 
shuffled for validation, the last rank is expected to see the padded samples. 
TODO: Remove dummy data from accuracy calculation
"""


def validate(val_loader, model, criterion, epoch, args):

    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()
    top1_count = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            start = time.time()

            if args.ipex:
                images = images.contiguous(memory_format=torch.channels_last)

            if args.ipex and args.bf16:
                images = images.to(torch.bfloat16)
                with torch.autocast(
                    "cpu",
                ):
                    output = model(images)
                output = output.to(torch.float32)
            if args.ipex and args.fp16:
                images = images.to(torch.half)
                with torch.autocast("cpu", dtype=torch.half):
                    output = model(images)
                output = output.to(torch.float32)
            else:
                output = model(images)

            loss = criterion(output, target)

            # measure accuracy and record loss
            acc, counts = accuracy(output, target, topk=(1, 5))
            acc1, acc5 = acc[0], acc[1]
            count1, _ = counts[0], counts[1]
            top1_count += count1.tolist()[0]
            batch_time.update(time.time() - start)
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            if i % args.print_freq == 0:
                progress.display(i)

    top1_count = torch.tensor(top1_count)
    if args.distributed:
        dist.barrier()
        dist.all_reduce(top1_count, op=dist.ReduceOp.SUM)
    top1_accuracy = top1_count.tolist() * (100.0 / 50000)
    print(
        "Validation top1 accuracy after epoch {epoch}: {top1} ".format(
            epoch=epoch, top1=top1_accuracy
        )
    )

    # batch_size = args.local_batch_size
    # latency = batch_time.avg / local_batch_size * 1000
    # perf = local_batch_size / batch_time.avg

    # print('inference latency %.3f ms'%latency)
    # print("Throughput: {:.3f} fps".format(perf))
    # print("Accuracy: {top1.avg:.3f} ".format(top1=top1))

    # # TODO: this should also be done with the ProgressMeter
    # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
    #         .format(top1=top1, top5=top5))

    return top1_accuracy


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
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
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        counts = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).int().sum(0, keepdim=True)
            res.append(correct_k.float().mul_(100.0 / batch_size))
            counts.append(correct_k)
        return res, counts


if __name__ == "__main__":
    main()
