#
# Copyright (c) 2023 Intel Corporation
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
# SPDX-License-Identifier: EPL-2.0
#

# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import json
import random
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from swim_transformer_data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, \
     NativeScalerWithGradNormCount, auto_resume_helper, reduce_tensor



def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--dummy', action='store_true', help='Use randomly generated dummy data. Low accuracy but no data required.')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    # for acceleration
    parser.add_argument('--fused_window_process', action='store_true',
                        help='Fused window shift & window partition, similar for reversed part.')
    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    ## overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')
    parser.add_argument('--plain', action='store_true', help='Generate plain format log files.')
    parser.add_argument('--num-iterations', type=int, default=20, help='Number of iterations to run')
    parser.add_argument('--device', type=str, default="xpu", choices=['xpu', 'cuda'])
    parser.add_argument('--profile', action='store_true', help='Get profiling info')
    parser.add_argument('--dtype', type=str, default="fp16", choices=["fp16", "tf32", "bf16", "fp32"])
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--max_epochs', type=int, help='max epoch number')
    parser.add_argument('--benchmark', action='store_true')
    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config, args.device)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    logger.info(str(model))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model, 'flops'):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    # TODO: go to xpu
    # model.cuda()
    model_without_ddp = model

    if args.device == "cuda":
        model = model.cuda()
    elif args.device == "xpu":
        model = model.xpu()

    optimizer = build_optimizer(config, model)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    # Gradient Scaling is only supported by cuda amp
    if args.device == "cuda":
        loss_scaler = NativeScalerWithGradNormCount()
    elif args.device == "xpu":
        loss_scaler = None

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.AUG.MIXUP > 0. and args.device != "xpu":
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0. and args.device != "xpu":
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if args.device == "xpu":
        criterion = criterion.xpu()

    max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger)
        assert args.num_iterations > 5, "The number of iterations should be great than 5"
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if config.EVAL_MODE:
            return

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model_without_ddp, logger)
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    model.train()

    # enable xpu.optimize
    if args.device == "xpu" and args.dtype == "tf32":
        model, optimizer = torch.xpu.optimize(model=model, optimizer=optimizer, dtype=torch.float32)
    if args.device == "xpu" and args.dtype == "bf16":
        model, optimizer = torch.xpu.optimize(model=model, optimizer=optimizer, dtype=torch.bfloat16)

    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        if hasattr(data_loader_train.sampler, 'set_epoch'):
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler,
                        loss_scaler, args.device, args.dtype)
        # if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
        if (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_scaler,
                            logger)

        # acc1, acc5, loss = validate(config, data_loader_val, model)
        # TODO add acc computation back, Zhiwei
        validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler, device, dtype):
    optimizer.zero_grad(set_to_none=True)

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()

    # original time measure for cuda
    start = time.time()
    end = time.time()

    warmup_iter = 5
    total_time_inner = 0
    for idx, (samples, targets) in enumerate(data_loader):
        with (torch.autograd.profiler_legacy.profile(enabled=args.profile, use_xpu=True) if args.device== "xpu"
            else torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]))  as prof:
            try:
                import memory_check
                memory_check.display_mem("xpu:0")
            except:
                pass
            start_time = time.time()
            # H2D
            if device == "cuda":
                samples = samples.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)
            elif device == "xpu":
                samples = samples.to("xpu")
                targets = targets.to("xpu")

            if mixup_fn is not None:
                samples, targets = mixup_fn(samples, targets)

            # forward
            if device == "cuda":
                with torch.autocast("cuda", enabled=config.AMP_ENABLE):
                    outputs = model(samples)
                loss = criterion(outputs, targets)
                loss = loss / config.TRAIN.ACCUMULATION_STEPS
            elif device == "xpu":
                if dtype == "bf16":
                    with torch.autocast("xpu", enabled=True, dtype=torch.bfloat16):
                        outputs = model(samples)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(samples)
                    loss = criterion(outputs, targets)
                loss = loss / config.TRAIN.ACCUMULATION_STEPS

            # cuda: with gradient scaling
            if device == "cuda":
                # this attribute is added by timm on one optimizer (adahessian)
                is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                        parameters=model.parameters(), create_graph=is_second_order,
                                        update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
                if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                    optimizer.zero_grad()
                    lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
                loss_scale_value = loss_scaler.state_dict()["scale"]
            # xpu: without gradient scaling
            elif device == "xpu":
                optimizer.zero_grad(set_to_none=True)
                # backward
                loss.backward()
                # gradient weight normalization
                if hasattr(optimizer, "clip_grad_norm_"):
                    ggnorm = optimizer.clip_grad_norm_(config.TRAIN.CLIP_GRAD)
                else:
                    ggnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                # weight update
                optimizer.step()

            # synchronize and measuring time
            if device == "cuda":
                torch.cuda.synchronize()
                loss_meter.update(loss.item(), targets.size(0))
                if grad_norm is not None:  # loss_scaler return None if not update
                    norm_meter.update(grad_norm)
                scaler_meter.update(loss_scale_value)
                batch_time.update(time.time() - end)
                end = time.time()

                if idx % config.PRINT_FREQ == 0:
                    lr = optimizer.param_groups[0]['lr']
                    wd = optimizer.param_groups[0]['weight_decay']
                    memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                    etas = batch_time.avg * (num_steps - idx)
                    logger.info(
                        f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                        f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                        f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                        f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                        f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                        f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                        f'mem {memory_used:.0f}MB')
            elif device == "xpu":
                torch.xpu.synchronize()
                end_time = time.time()
                loss_meter.update(loss.item(), targets.size(0))
                batch_time.update(time.time() - end)

                if idx % config.PRINT_FREQ == 0:
                    lr = optimizer.param_groups[0]['lr']
                    wd = optimizer.param_groups[0]['weight_decay']
                    etas = batch_time.avg * (num_steps - idx)
                    logger.info(
                        f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                        f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                        f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                        f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t')

        if args.profile and idx == 450:
            title = "/swin_transform_training_"
            title += args.dtype + "_"
            title += "bs" + str(config.DATA.BATCH_SIZE) + "_"
            title = (title + "plain_") if args.plain else (title + "block_")
            title += args.device + "_"

            with open('./' + title + 'profile.txt', 'w') as f:
                f.write(str(prof.key_averages().table(sort_by="self_xpu_time_total", row_limit=100000)))
            with open('./' + title + 'profile_detailed.txt', 'w') as f1:
                f1.write(str(prof.table(sort_by="id", row_limit=100000)))
            prof.export_chrome_trace('./' + title + 'profile_trace.json')

        if idx >= warmup_iter:  # warmup iterations
            total_time_inner += end_time - start_time
            latency = total_time_inner / (idx - warmup_iter + 1)
            throughput = config.DATA.BATCH_SIZE / latency
            if idx == 500:
                print("---latency={} ms".format(latency))
                print("---throughput={} fps".format(throughput))
                exit()


    if device == "cuda":
        epoch_time = time.time() - start
        logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


def write_profile_info(prof, bs):
    title = "./swin_transform_inference_"
    title += "fp16" + "_"
    title += "bs" + str(bs) + "_"
    title = (title + "plain_") if args.plain else (title + "block_")
    title += args.device + "_"

    profiling_path = os.getenv('PROFILE_PATH')
    if not profiling_path:
        profiling_path = ''
    sort_str = "self_" + args.device + "_time_total"
    torch.save(prof.key_averages().table(sort_by=sort_str), profiling_path + title + 'profiling.pt')
    torch.save(prof.key_averages(group_by_input_shape=True).table(), profiling_path + title + 'profiling_detailed.pt')
    prof.export_chrome_trace(profiling_path + title + 'profiling.json')
    print(prof.key_averages().table(sort_by=sort_str))
    print(prof.key_averages().table(sort_by=sort_str), file=open(profiling_path + title + '.breakdown', 'w'))
    print(prof.key_averages(group_by_input_shape=True).table())

def print_perf_info(total_time, counter, batch_size):
    latency = total_time / counter
    throughput = batch_size / latency
    logger.info(f'Latency: {latency:.6f}')
    logger.info(f'Throughput: {throughput:.6f}')
    # logger.info(
    #             f'Test: [{idx}/{len(data_loader)}]\t'
    #             f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
    #             f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    model = model.to(args.device)
    # TODO: move to a better position
    # model = model.half()
    # OOB solution
    model = torch.xpu.optimize(model=model, dtype=torch.float16)

    print("***** start tracing *****")
    with torch.no_grad():
        with torch.autocast("xpu", enabled=True, dtype=torch.float16):
            model = torch.jit.trace(model, torch.randn(1, 3, 224, 224).to(args.device).half())
    print("***** finish tracing *****")

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()

    profiling_enabled = False
    total_time = 0.0
    counter = 0
    if args.num_iterations > len(data_loader):
        args.benchmark=True
    total_idx = 0
    while total_idx < args.num_iterations:
        for idx, (images, target) in enumerate(data_loader):
            # TODO: Move to cpu
            profiling_enabled = args.profile
            batch_size = args.batch_size
            with (torch.autograd.profiler_legacy.profile(enabled=profiling_enabled, use_xpu=True, record_shapes=False) if args.device== "xpu"
                else torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]))  as prof:
                try:
                    import memory_check
                    memory_check.display_mem("xpu:0")
                except:
                    pass
                start_time = time.time()
                # H2D
                images = images.half().to(args.device)
                target = target.to(args.device)
                # Forward
                with torch.no_grad():
                    output = model(images)
                # D2H
                output = output.cpu()
                # Synchronize
                if args.device=="xpu":
                    torch.xpu.synchronize()
                else:
                    torch.cuda.synchronize()
                end_time = time.time()

            if (idx > 5) and (images.size(0)==args.batch_size): # Avoid compute last batch of dataset
                total_time += (end_time - start_time)
                counter += 1
            if profiling_enabled and idx == 10:
                write_profile_info(prof, batch_size)
            if total_idx == args.num_iterations -1:
                print_perf_info(total_time, counter, batch_size)
                exit()
            # measure accuracy and record loss
            output = output.to(args.device)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            acc1 = reduce_tensor(acc1)
            acc5 = reduce_tensor(acc5)
            loss = reduce_tensor(loss)

            loss_meter.update(loss.item(), target.size(0))
            acc1_meter.update(acc1.item(), target.size(0))
            acc5_meter.update(acc5.item(), target.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % config.PRINT_FREQ == 0:
                # TODO: Add xpu version
                # memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                logger.info(
                    f'Test: [{idx}/{len(data_loader)}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                    f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t')
                    # f'Mem {memory_used:.0f}MB')
            total_idx += 1
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    args, config = parse_option()

    if args.device == "xpu":
        import intel_extension_for_pytorch

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RANK'] = str(os.environ.get('PMI_RANK', -1))
    os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', -1))
    os.environ['CCL_RANK'] = os.environ['RANK']
    os.environ['CCL_SIZE'] = os.environ['WORLD_SIZE']
    args.profile = os.environ.get("PROFILE", "OFF").upper() in ["1", "Y", "ON", "YES", "TRUE"]

    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1


    seed = config.SEED
    torch.manual_seed(seed)
    if args.device == "cuda":
        torch.cuda.manual_seed(seed)
    elif args.device == "xpu":
        torch.xpu.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    #cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    # linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE  / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE  / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    # TODO: Zhiwei, dist_rank is set to 0, need distribution support
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.NAME}")

    # if dist.get_rank() == 0:
    if True:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    main(config)
