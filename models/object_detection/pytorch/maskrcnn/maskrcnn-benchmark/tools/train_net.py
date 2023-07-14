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
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config
import intel_extension_for_pytorch as ipex


def train(cfg, local_rank, distributed, bf16=False, bf32=False, iterations=-1, iter_warmup=-1):
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    model = model.to(memory_format=torch.channels_last)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    if bf32:
        ipex.set_fp32_math_mode(mode=ipex.FP32MathMode.BF32, device="cpu")
        model, optimizer = ipex.optimize(model, dtype=torch.float32, optimizer=optimizer, inplace=True, auto_kernel_selection=True)
    elif bf16:
        model, optimizer = ipex.optimize(model, dtype=torch.bfloat16, optimizer=optimizer, inplace=True)
    else:
        model, optimizer = ipex.optimize(model, dtype=torch.float32, optimizer=optimizer, inplace=True)

    if distributed:
        device_ids = None
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=device_ids, find_unused_parameters=True)

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )

    test_period = cfg.SOLVER.TEST_PERIOD
    if test_period > 0:
        data_loader_val = make_data_loader(cfg, is_train=False, is_distributed=distributed, is_for_period=True)
    else:
        data_loader_val = None

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    do_train(
        cfg,
        model,
        data_loader,
        data_loader_val,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        test_period,
        arguments,
        bf16=bf16,
        bf32=bf32,
        iterations=iterations,
        iter_warmup=iter_warmup
    )

    return model


def run_test(cfg, model, distributed):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            ims_per_patch=cfg.TEST.IMS_PER_BATCH,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            bbox_aug=cfg.TEST.BBOX_AUG.ENABLED,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument('--bf16', action='store_true', default=False,
                        help='enable BF16 by IPEX autocast')
    parser.add_argument('--bf32', action='store_true', default=False,
                        help='enable IPEX bf32 path')
    parser.add_argument('-i', '--iterations', default=-1, type=int, metavar='N',
                        help='number of total iterations to run')
    parser.add_argument('--iter-warmup', default=-1, type=int, metavar='N',
                        help='number of warm-up iterations to run')
    parser.add_argument('-b', '--local-batch-size', default=1, type=int, metavar='N',
                        help='local batch size')
    parser.add_argument("--world-size", default=1, type=int, help='world size')
    parser.add_argument("--master-addr", default='127.0.0.1', type=str, help='Master Addr')
    parser.add_argument("--port", default='29500', type=str, help='Port')
    parser.add_argument("--rank", default=0, type=int, help='rank')
    parser.add_argument('--backend', default='gloo', type=str, help='DDP backend, default to gloo')

    args = parser.parse_args()

    args.distributed = False
    batch_size = args.local_batch_size
    if torch.distributed.is_available() and int(os.environ.get('PMI_SIZE', '0')) > 1:
        print('Distributed training with DDP')
        os.environ['RANK'] = os.environ.get('PMI_RANK', '0')
        os.environ['WORLD_SIZE'] = os.environ.get('PMI_SIZE', '1')
        if not 'MASTER_ADDR' in os.environ:
            os.environ['MASTER_ADDR'] = args.master_addr
        if not 'MASTER_PORT' in os.environ:
            os.environ['MASTER_PORT'] = args.port

        # Initialize the process group with ccl backend
        if args.backend == 'ccl':
            if torch.__version__[:6] >= '1.12.0':
                import oneccl_bindings_for_pytorch
            else:
                import torch_ccl
        torch.distributed.init_process_group(
                backend=args.backend                
        )
        args.distributed = True
        if torch.distributed.is_initialized():
            print("Torch distributed is initialized.")
            args.rank = torch.distributed.get_rank()
            args.world_size = torch.distributed.get_world_size()
        else:
            print("Torch distributed is not initialized.")
            args.rank = 0
            args.world_size = 1
        batch_size = args.local_batch_size * args.world_size
        print("Rank and world size: ", args.rank," ", args.world_size)
        print("Using local batch size: ", args.local_batch_size)
        print("Using global batch size: ", batch_size)

    cfg.merge_from_file(args.config_file)
    args.opts.append("SOLVER.IMS_PER_BATCH")
    args.opts.append(batch_size)
    if "SOLVER.STEPS_1" in args.opts and \
        "SOLVER.STEPS_2" in args.opts and \
        "SOLVER.STEPS" not in args.opts:
        steps_1_ind = args.opts.index("SOLVER.STEPS_1") + 1
        steps_2_ind = args.opts.index("SOLVER.STEPS_2") + 1
        new_steps = ["SOLVER.STEPS", '({},{})'.format(args.opts[steps_1_ind], args.opts[steps_2_ind])]
        print(f"Combine SOLVER.STEPS_1 and SOLVER.STEPS_2 together to generate {new_steps}")
        steps_1_key_ind = args.opts.index("SOLVER.STEPS_1")
        args.opts.pop(steps_1_key_ind)
        args.opts.pop(steps_1_key_ind)
        steps_2_key_ind = args.opts.index("SOLVER.STEPS_2")
        args.opts.pop(steps_2_key_ind)
        args.opts.pop(steps_2_key_ind)
        print("Remove SOLVER.STEPS_1 and SOLVER.STEPS_2 from args.opt")
        args.opts += new_steps
        print(args.opts)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    model = train(cfg, args.local_rank, args.distributed,
                  bf16=args.bf16, bf32=args.bf32, iterations=args.iterations, iter_warmup=args.iter_warmup)

    if not args.skip_test:
        run_test(cfg, model, args.distributed)


if __name__ == "__main__":
    main()
