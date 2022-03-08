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

### This file is originally from: [mlcommons repo](https://github.com/mlcommons/training/tree/9947bdf21ee3f2488fa4b362eec2ce7deb2ec4dd/single_stage_detector/ssd/train.py)
import os
from argparse import ArgumentParser
from utils import DefaultBoxes, Encoder, COCODetection
from base_model import Loss
from utils import SSDTransformer
from ssd300 import SSD300
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import time
import random
import numpy as np
import logging
from mlperf_logging.mllog import constants as mllog_const
from mlperf_logger import ssd_print, broadcast_seeds
from mlperf_logger import mllogger

use_ipex = False
if os.environ.get('USE_IPEX') == "1":
    import intel_extension_for_pytorch as ipex
    use_ipex = True

_BASE_LR=2.5e-3

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

def trace_handler(prof):
    print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    torch.profiler.tensorboard_trace_handler('./ssdrn34_train_log')(prof)

def parse_args():
    parser = ArgumentParser(description="Train Single Shot MultiBox Detector"
                                        " on COCO")
    parser.add_argument('--data', '-d', type=str, default='/coco',
                        help='path to test and training data files')
    parser.add_argument('--pretrained-backbone', type=str, default=None,
                        help='path to pretrained backbone weights file, '
                             'default is to get it from online torchvision repository')
    parser.add_argument('--epochs', '-e', type=int, default=800,
                        help='number of epochs for training')
    parser.add_argument('--batch-size', '-b', type=int, default=32,
                        help='number of examples for each training iteration')
    parser.add_argument('--val-batch-size', type=int, default=None,
                        help='number of examples for each validation iteration (defaults to --batch-size)')
    parser.add_argument('--no_cuda', action='store_true', default=True,
                        help='use available GPUs')
    parser.add_argument('--seed', '-s', type=int, default=random.SystemRandom().randint(0, 2**32 - 1),
                        help='manually set random seed for torch')
    parser.add_argument('--threshold', '-t', type=float, default=0.23,
                        help='stop training early at threshold')
    parser.add_argument('--iteration', type=int, default=0,
                        help='iteration to start from')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='path to model checkpoint file')
    parser.add_argument('--no-save', action='store_true',
                        help='save model checkpoints')
    parser.add_argument('--val-interval', type=int, default=5,
                        help='epoch interval for validation in addition to --val-epochs.')
    parser.add_argument('--val-epochs', nargs='*', type=int,
                        default=[],
                        help='epochs at which to evaluate in addition to --val-interval')
    parser.add_argument('--batch-splits', type=int, default=1,
                        help='Split batch to N steps (gradient accumulation)')
    parser.add_argument('--lr-decay-schedule', nargs='*', type=int,
                        default=[40, 50],
                        help='epochs at which to decay the learning rate')
    parser.add_argument('--warmup', type=float, default=None,
                        help='how long the learning rate will be warmed up in fraction of epochs')
    parser.add_argument('--warmup-factor', type=int, default=0,
                        help='mlperf rule parameter for controlling warmup curve')
    parser.add_argument('--lr', type=float, default=_BASE_LR,
                        help='base learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='weight decay factor')
    parser.add_argument('--num-cropping-iterations', type=int, default=1,
                        help='cropping retries in augmentation pipeline, '
                             'default 1, other legal value is 50')
    parser.add_argument('--nms-valid-thresh', type=float, default=0.05,
                        help='in eval, filter input boxes to those with score greater '
                             'than nms_valid_thresh.')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Logging mini-batch interval.')
    # Distributed stuff
    parser.add_argument('--distributed', action='store_true', default=False, help='enable distributed')
    parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', 0), type=int,
                        help='Used for multi-process training. Can either be manually set '
                             'or automatically set by using \'python -m multiproc\'.')
    parser.add_argument('--rank', default=0, type=int, help='rank')
    parser.add_argument('--local_world_size', default=1, type=int, help='local world size')
    parser.add_argument('--world_size', default=1, type=int, help='world size')
    parser.add_argument('--backend', default='gloo', type=str, help='DDP backend, default to gloo')
    parser.add_argument('--master_addr', default='127.0.0.1', type=str, help='Master Addr')
    parser.add_argument('--port', default='29500', type=str, help='Port')
    # Added for performance measurement
    parser.add_argument('--performance_only', action='store_true', default=False,
                        help='only for performance test')
    parser.add_argument('-w', '--warmup-iterations', default=0, type=int, metavar='N',
                        help='number of warmup iterations to run')
    parser.add_argument('-iter', '--train-iteration', type=int, default=None,
                        help='set the iteration for the performance test of train, default is None')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    # Added for BF16 training
    parser.add_argument('--autocast', action='store_true', default=False,
                        help='enable autocast')
    parser.add_argument('--profile', action='store_true', default=False,
                        help='enable profile')
    return parser.parse_args()


def show_memusage(device=0):
    import gpustat
    gpu_stats = gpustat.GPUStatCollection.new_query()
    item = gpu_stats.jsonify()["gpus"][device]
    print("{}/{}".format(item["memory.used"], item["memory.total"]))


def dboxes300_coco():
    figsize = 300
    feat_size = [38, 19, 10, 5, 3, 1]
    steps = [8, 16, 32, 64, 100, 300]
    # use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
    scales = [21, 45, 99, 153, 207, 261, 315]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    return dboxes


def coco_eval(model, val_dataloader, cocoGt, encoder, inv_map, threshold,
              epoch, iteration, log_interval=100,
              use_cuda=False, nms_valid_thresh=0.05, use_autocast=False):
    from pycocotools.cocoeval import COCOeval
    print("")
    model.eval()
    if use_cuda:
        model.cuda()
    ret = []

    overlap_threshold = 0.50
    nms_max_detections = 200
    print("nms_valid_thresh is set to {}".format(nms_valid_thresh))

    mllogger.start(
        key=mllog_const.EVAL_START,
        metadata={mllog_const.EPOCH_NUM: epoch})

    start = time.time()
    for nbatch, (img, img_id, img_size, bbox, label) in enumerate(val_dataloader):
        with torch.no_grad():
            if use_cuda:
                img = img.cuda()
            # img to nhwc
            img = img.contiguous(memory_format=torch.channels_last)
            if use_autocast:
                with torch.cpu.amp.autocast(enabled=use_autocast):
                    ploc, plabel = model(img)
                ploc = ploc.to(torch.float32)
                plabel = plabel.to(torch.float32)
            else:
                ploc, plabel = model(img)

            try:
                results = encoder.decode_batch(ploc, plabel,
                                               overlap_threshold,
                                               nms_max_detections,
                                               nms_valid_thresh=nms_valid_thresh)
            except:
                #raise
                print("")
                print("No object detected in batch: {}".format(nbatch))
                continue

            (htot, wtot) = [d.cpu().numpy() for d in img_size]
            img_id = img_id.cpu().numpy()
            # Iterate over batch elements
            for img_id_, wtot_, htot_, result in zip(img_id, wtot, htot, results):
                loc, label, prob = [r.cpu().numpy() for r in result]

                # Iterate over image detections
                for loc_, label_, prob_ in zip(loc, label, prob):
                    ret.append([img_id_, loc_[0]*wtot_, \
                                         loc_[1]*htot_,
                                         (loc_[2] - loc_[0])*wtot_,
                                         (loc_[3] - loc_[1])*htot_,
                                         prob_,
                                         inv_map[label_]])
        if log_interval and not (nbatch+1) % log_interval:
                print("Completed inference on batch: {}".format(nbatch+1))

    print("")
    print("Predicting Ended, total time: {:.2f} s".format(time.time()-start))

    cocoDt = cocoGt.loadRes(np.array(ret))

    E = COCOeval(cocoGt, cocoDt, iouType='bbox')
    E.evaluate()
    E.accumulate()
    E.summarize()
    print("Current AP: {:.5f} AP goal: {:.5f}".format(E.stats[0], threshold))

    # put your model back into training mode
    model.train()

    current_accuracy = E.stats[0]

    ssd_print(key=mllog_const.EVAL_ACCURACY,
              value=current_accuracy,
              metadata={mllog_const.EPOCH_NUM: epoch},
              sync=False)
    mllogger.end(
        key=mllog_const.EVAL_STOP,
        metadata={mllog_const.EPOCH_NUM: epoch})
    return current_accuracy >= threshold #Average Precision  (AP) @[ IoU=050:0.95 | area=   all | maxDets=100 ]

def lr_warmup(optim, wb, iter_num, base_lr, args):
	if iter_num < wb:
		# mlperf warmup rule
		warmup_step = base_lr / (wb * (2 ** args.warmup_factor))
		new_lr = base_lr - (wb - iter_num) * warmup_step

		for param_group in optim.param_groups:
			param_group['lr'] = new_lr

def data_preprocess(img, bbox, label, loss_func, use_autocast):
    trans_bbox = bbox.transpose(1,2).contiguous()
    # image to NHWC
    img = img.contiguous(memory_format=torch.channels_last)
    if use_autocast:
        trans_bbox = loss_func._loc_vec(trans_bbox).to(torch.bfloat16)
    else:
        trans_bbox = loss_func._loc_vec(trans_bbox).to(torch.float32)
    mask = label > 0
    pos_num = mask.sum(dim=1)
    neg_num = torch.clamp(3*pos_num, max=mask.size(1)).unsqueeze(-1)
    num_mask = (pos_num > 0).float()
    return (img, trans_bbox, label, mask, pos_num, neg_num, num_mask)

def train300_mlperf_coco(args):
    global torch
    from coco import COCO
    # Check that GPUs are actually available
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.distributed = False
    if use_cuda:
        try:
            from apex.parallel import DistributedDataParallel as DDP
            if 'WORLD_SIZE' in os.environ:
                args.distributed = int(os.environ['WORLD_SIZE']) > 1
        except:
            raise ImportError("Please install APEX from https://github.com/nvidia/apex")

    local_seed = args.seed
    os.environ['USE_CUDA'] = str(use_cuda)
    if args.world_size > 1:
        args.distributed = True

    if args.distributed:
        # necessary pytorch imports
        import torch.utils.data.distributed
        import torch.distributed as dist
        print('Distributed training with DDP')
        if args.no_cuda:
            device = torch.device('cpu')
            os.environ['RANK'] = str(os.environ.get('PMI_RANK', args.rank))
            os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', args.world_size))
            os.environ['MASTER_ADDR'] = args.master_addr
            os.environ['MASTER_PORT'] = args.port

            # Initialize the process group with ccl backend
            if args.backend == 'ccl':
                import torch_ccl
            dist.init_process_group(
                backend=args.backend
            )
        else:
            torch.cuda.set_device(args.local_rank)
            device = torch.device('cuda')
            dist.init_process_group(backend='nccl',
                                    init_method='env://')
            # set seeds properly
            args.seed = broadcast_seeds(args.seed, device)
            local_seed = (args.seed + dist.get_rank()) % 2**32
    mllogger.event(key=mllog_const.SEED, value=local_seed)
    # Refer to https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    torch.manual_seed(local_seed) # Set PyTorch seed
    np.random.seed(seed=local_seed) # Set Numpy seed
    random.seed(local_seed) # Set the Python seed

    args.rank = dist.get_rank() if args.distributed else args.local_rank
    print("args.rank = {}".format(args.rank))
    print("local rank = {}".format(args.local_rank))
    print("distributed={}".format(args.distributed))

    dboxes = dboxes300_coco()
    encoder = Encoder(dboxes)

    input_size = 300
    train_trans = SSDTransformer(dboxes, (input_size, input_size), val=False,
                                 num_cropping_iterations=args.num_cropping_iterations)
    val_trans = SSDTransformer(dboxes, (input_size, input_size), val=True)

    val_annotate = os.path.join(args.data, "annotations/instances_val2017.json")
    val_coco_root = os.path.join(args.data, "val2017")
    train_annotate = os.path.join(args.data, "annotations/instances_train2017.json")
    train_coco_root = os.path.join(args.data, "train2017")

    cocoGt = COCO(annotation_file=val_annotate)
    train_coco = COCODetection(train_coco_root, train_annotate, train_trans)
    val_coco = COCODetection(val_coco_root, val_annotate, val_trans)
    mllogger.event(key=mllog_const.TRAIN_SAMPLES, value=len(train_coco))
    mllogger.event(key=mllog_const.EVAL_SAMPLES, value=len(val_coco))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_coco)
    else:
        train_sampler = None
    train_dataloader = DataLoader(train_coco,
                                  batch_size=args.batch_size,
                                  shuffle=(train_sampler is None),
                                  sampler=train_sampler,
                                  num_workers=0)
    # set shuffle=True in DataLoader
    # Leslie: here is the workaround: dist.broadcast will fail on other rank. we will run evalution on all the ranks
    val_dataloader = DataLoader(val_coco,
                                batch_size=args.val_batch_size or args.batch_size,
                                shuffle=False,
                                sampler=None,
                                num_workers=0)

    ssd300 = SSD300(train_coco.labelnum, model_path=args.pretrained_backbone)

    ssd300.train()
    if use_cuda:
        ssd300.cuda()
    loss_func = Loss(dboxes)
    if use_cuda:
        loss_func.cuda()
    if args.distributed:
        N_gpu = torch.distributed.get_world_size()
    else:
        N_gpu = 1

    global_batch_size = N_gpu * args.batch_size
    mllogger.event(key=mllog_const.GLOBAL_BATCH_SIZE, value=global_batch_size)
    # Reference doesn't support group batch norm, so bn_span==local_batch_size
    mllogger.event(key=mllog_const.MODEL_BN_SPAN, value=args.batch_size)
    current_lr = args.lr * (global_batch_size / 32)

    assert args.batch_size % args.batch_splits == 0, "--batch-size must be divisible by --batch-splits"
    fragment_size = args.batch_size // args.batch_splits
    if args.batch_splits != 1:
        print("using gradient accumulation with fragments of size {}".format(fragment_size))

    # Model to NHWC
    ssd300 = ssd300.to(memory_format=torch.channels_last)

    current_momentum = 0.9
    optim = torch.optim.SGD(ssd300.parameters(), lr=current_lr,
                            momentum=current_momentum,
                            weight_decay=args.weight_decay)
    ssd_print(key=mllog_const.OPT_BASE_LR, value=current_lr)
    ssd_print(key=mllog_const.OPT_WEIGHT_DECAY, value=args.weight_decay)

    iter_num = args.iteration
    avg_loss = 0.0
    inv_map = {v:k for k,v in val_coco.label_map.items()}
    success = torch.zeros(1)
    if use_cuda:
        success = success.cuda()


    if args.warmup:
        nonempty_imgs = len(train_coco)
        wb = int(args.warmup * nonempty_imgs / (N_gpu*args.batch_size))
        ssd_print(key=mllog_const.OPT_LR_WARMUP_STEPS, value=wb)
        warmup_step = lambda iter_num, current_lr: lr_warmup(optim, wb, iter_num, current_lr, args)
    else:
        warmup_step = lambda iter_num, current_lr: None

    ssd_print(key=mllog_const.OPT_LR_WARMUP_FACTOR, value=args.warmup_factor)
    ssd_print(key=mllog_const.OPT_LR_DECAY_BOUNDARY_EPOCHS, value=args.lr_decay_schedule)
    mllogger.start(
        key=mllog_const.BLOCK_START,
        metadata={mllog_const.FIRST_EPOCH_NUM: 1,
                  mllog_const.EPOCH_COUNT: args.epochs})

    if args.performance_only:
        train_time = AverageMeter('TrainTime', ':6.3f')
        progress = ProgressMeter(
            args.train_iteration,
            [train_time],
            prefix='Train: ')

    # Restore the model and optim from checkpoint
    if args.checkpoint is not None:
        print("loading model checkpoint", args.checkpoint)
        od = torch.load(args.checkpoint)
        ssd300.load_state_dict(od["model"])
        optim.load_state_dict(od['optim'])

    # Model Prepack
    if use_ipex:
        if args.autocast:
            ssd300, optim = ipex.optimize(ssd300, dtype=torch.bfloat16, optimizer=optim)
        else:
            ssd300, optim = ipex.optimize(ssd300, dtype=torch.float32, optimizer=optim)

    # parallelize
    if args.distributed:
        device_ids = None
        ssd300 = torch.nn.parallel.DistributedDataParallel(ssd300, device_ids=device_ids)

    optim.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        mllogger.start(
            key=mllog_const.EPOCH_START,
            metadata={mllog_const.EPOCH_NUM: epoch})
        # set the epoch for the sampler
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if epoch in args.lr_decay_schedule:
            current_lr *= 0.1
            print("")
            print("lr decay step #{num}".format(num=args.lr_decay_schedule.index(epoch) + 1))
            for param_group in optim.param_groups:
                param_group['lr'] = current_lr
        for nbatch, (img, img_id, img_size, bbox, label) in enumerate(train_dataloader):
            naive_train_case = True # img.shape[0] == fragment_size
            if naive_train_case:
                # Naive train case
                fimg, gloc, glabel, mask, pos_num, neg_num, num_mask = data_preprocess(img, bbox, label, loss_func, args.autocast)

                if args.performance_only and iter_num >= args.warmup_iterations:
                    start_time = time.time()
                if args.profile and args.performance_only and iter_num == 30:
                    # Profile Mode
                    with torch.profiler.profile(on_trace_ready=trace_handler) as prof:
                        with torch.cpu.amp.autocast(enabled=args.autocast):
                            ploc, plabel = ssd300(fimg)
                            loss = loss_func(ploc, plabel, gloc, glabel, mask, pos_num, neg_num, num_mask, args.autocast)
                        loss.backward()

                        warmup_step(iter_num, current_lr)
                        optim.step()
                        optim.zero_grad(set_to_none=True)
                else:
                    # Non Profile Mode
                    with torch.cpu.amp.autocast(enabled=args.autocast):
                        ploc, plabel = ssd300(fimg)
                        loss = loss_func(ploc, plabel, gloc, glabel, mask, pos_num, neg_num, num_mask, args.autocast)
                    loss.backward()

                    warmup_step(iter_num, current_lr)
                    optim.step()
                    optim.zero_grad(set_to_none=True)
            else:
                # Train case: when split input to several fragment size
                print("Not support input with several fragment size yet.")
                exit(-1)
                # current_batch_size = img.shape[0]
                # # Split batch for gradient accumulation
                # img = torch.split(img, fragment_size)
                # bbox = torch.split(bbox, fragment_size)
                # label = torch.split(label, fragment_size)

                # if args.performance_only and iter_num >= args.warmup_iterations:
                #     start_time=time.time()
                # for (fimg, fbbox, flabel) in zip(img, bbox, label):
                #     current_fragment_size = fimg.shape[0]
                #     trans_bbox = fbbox.transpose(1,2).contiguous()
                #     if use_cuda:
                #         fimg = fimg.cuda()
                #         trans_bbox = trans_bbox.cuda()
                #         flabel = flabel.cuda()
                #     fimg = Variable(fimg, requires_grad=True)
                #     gloc, glabel = Variable(trans_bbox, requires_grad=False), \
                #                 Variable(flabel, requires_grad=False)
                #     gloc = loss_func._loc_vec(gloc)
                #     mask = glabel > 0
                #     pos_num = mask.sum(dim=1)
                #     neg_num = torch.clamp(3*pos_num, max=mask.size(1)).unsqueeze(-1)
                #     num_mask = (pos_num > 0).float()
                #     # image to NHWC
                #     fimg = fimg.contiguous(memory_format=torch.channels_last)
                #     if use_ipex:
                #         with ipex.amp.autocast(enabled=args.autocast, configure=ipex.conf.AmpConf(torch.bfloat16)):
                #             ploc, plabel = ssd300(fimg)
                #             loss = loss_func(ploc, plabel, gloc, glabel, mask, pos_num, neg_num, num_mask)
                #     else:
                #         ploc, plabel = ssd300(fimg)
                #         loss = loss_func(ploc, plabel, gloc, glabel, mask, pos_num, neg_num, num_mask)
                #     loss = loss * (current_fragment_size / current_batch_size) # weighted mean
                #     loss.backward()

                # warmup_step(iter_num, current_lr)
                # optim.step()
                # optim.zero_grad(set_to_none=True)
            if args.performance_only and iter_num >= args.warmup_iterations:
                train_time.update(time.time() - start_time)
            if args.performance_only and iter_num % args.print_freq == 0:
                progress.display(iter_num)
            if not np.isinf(loss.item()): avg_loss = 0.999*avg_loss + 0.001*loss.item()
            if args.log_interval and not iter_num % args.log_interval:
                print("Iteration: {:6d}, Loss function: {:5.8f}, Average Loss: {:.8f}"\
                    .format(iter_num, loss.item(), avg_loss))
            iter_num += 1
            if args.performance_only and iter_num >= args.train_iteration:
                break
        if args.performance_only and iter_num >= args.train_iteration:
            break

        if (args.val_epochs and (epoch+1) in args.val_epochs) or \
           (args.val_interval and not (epoch+1) % args.val_interval):
            if args.distributed:
                world_size = float(dist.get_world_size())
                for bn_name, bn_buf in ssd300.module.named_buffers(recurse=True):
                    if ('running_mean' in bn_name) or ('running_var' in bn_name):
                        dist.all_reduce(bn_buf, op=dist.ReduceOp.SUM)
                        bn_buf /= world_size
                        ssd_print(key=mllog_const.MODEL_BN_SPAN,
                            value=bn_buf.cpu().detach().numpy())
            if args.rank == 0 or True: # Leslie: here is the workaround: dist.broadcast will fail on other rank. we will run evalution on all the ranks
                if not args.no_save:
                    print("")
                    print("saving model...")
                    torch.save({"model" : ssd300.state_dict(), "label_map": train_coco.label_info, "optim": optim.state_dict()},
                               "./models/iter_{}.pt".format(iter_num))

                if coco_eval(ssd300, val_dataloader, cocoGt, encoder, inv_map,
                             args.threshold, epoch + 1, iter_num,
                             log_interval=args.log_interval,
                             nms_valid_thresh=args.nms_valid_thresh, use_autocast=args.autocast):
                    success = torch.ones(1)
                    if use_cuda:
                        success = success.cuda()
            # Leslie: same Workaround: since we run evalution on all ranks, we don't need to broadcast the evalutation result
            # if args.distributed:
            #     dist.broadcast(success, 0)
            if success[0]:
                return True
            mllogger.end(
                key=mllog_const.EPOCH_STOP,
                metadata={mllog_const.EPOCH_NUM: epoch})
    mllogger.end(
        key=mllog_const.BLOCK_STOP,
        metadata={mllog_const.FIRST_EPOCH_NUM: 1,
                  mllog_const.EPOCH_COUNT: args.epochs})

    if args.performance_only:
        batch_size = args.batch_size
        latency = train_time.avg / batch_size * 1000
        perf = batch_size / train_time.avg
        print('train latency %.2f ms'%latency)
        print('train performance %.2f fps'%perf)
        print("Throughput: {:.3f} fps".format(perf))

    return False

def main():
    mllogger.start(key=mllog_const.INIT_START)
    args = parse_args()
    print(args)

    if args.local_rank == 0:
        if not os.path.isdir('./models'):
            os.mkdir('./models')

    torch.backends.cudnn.benchmark = True

    # start timing here
    mllogger.end(key=mllog_const.INIT_STOP)
    mllogger.start(key=mllog_const.RUN_START)

    success = train300_mlperf_coco(args)

    # end timing here
    mllogger.end(key=mllog_const.RUN_STOP, value={"success": success})


if __name__ == "__main__":
    main()
