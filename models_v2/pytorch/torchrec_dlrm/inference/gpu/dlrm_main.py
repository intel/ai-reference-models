#!/usr/bin/env python3

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
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import contextlib
import itertools
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
import numpy as np

import torch
import torchmetrics as metrics
from torch.autograd.profiler import record_function
from pyre_extensions import none_throws
from torch import distributed as dist
from torch.utils.data import DataLoader
from torchrec import EmbeddingBagCollection
from torchrec.datasets.criteo import DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES
#from torchrec.distributed import TrainPipelineSparseDist
from torchrec.distributed.comm import get_local_size
#from torchrec.distributed.model_parallel import (
#    DistributedModelParallel,
#    get_default_sharders,
#)
#from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
#from torchrec.distributed.planner.storage_reservations import (
#    HeuristicalStorageReservation,
#)
from dist_models import DIST_DLRM_DCN, replace_embeddingbag_collection
from torchrec.models.dlrm import DLRM, DLRM_DCN, DLRM_Projection, DLRMTrain
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter
from tqdm import tqdm
import extend_distributed as ext_dist
from extend_distributed import dist, env2int
from jit_trace_able_utils import unpack, SparseArchTraceAbleWrapper

import logging
import sharding

logging.basicConfig(level=logging.DEBUG)
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(1)

# OSS import
try:
    # pyre-ignore[21]
    # @manual=//ai_codesign/benchmarks/dlrm/torchrec_dlrm/data:dlrm_dataloader
    from data.dlrm_dataloader import get_dataloader

    # pyre-ignore[21]
    # @manual=//ai_codesign/benchmarks/dlrm/torchrec_dlrm:lr_scheduler
    from lr_scheduler import LRPolicyScheduler

    # pyre-ignore[21]
    # @manual=//ai_codesign/benchmarks/dlrm/torchrec_dlrm:multi_hot
    from multi_hot import Multihot, RestartableMap
except ImportError:
    pass

# internal import
try:
    from .data.dlrm_dataloader import get_dataloader  # noqa F811
    from .lr_scheduler import LRPolicyScheduler  # noqa F811
    from .multi_hot import Multihot, RestartableMap  # noqa F811
except ImportError:
    pass

TRAIN_PIPELINE_STAGES = 3  # Number of stages in TrainPipelineSparseDist.

n_emb = None
#local_emb_slice = None
local_emb_indices = None
default_embedding_names = None
idx_name = None


def load_checkoutpoint(args, device):
    checkpoint = None
    if args.checkpoint_floder is not None:
        if (not os.path.exists(args.checkpoint_floder)) or (not os.path.isdir(args.checkpoint_floder)):
            logger.error("The checkpoint dir is not exist.")
            exit()
        if  not os.path.exists(args.checkpoint_floder + '/dlrm_training_' + str(dist.get_rank()) + '_save.pt'):
            logger.warning("The checkpoint " + args.checkpoint_floder + '/dlrm_training_' + str(dist.get_rank()) + '_save.pt' +  " doesn't exist. If converge isn't zero step, your behavior will cause exceptions.")
        else:
            print('load...', args.checkpoint_floder + '/dlrm_training_' + str(dist.get_rank()) + '_save.pt', ' to ',
                  device)
            checkpoint = torch.load(args.checkpoint_floder + '/dlrm_training_' + str(dist.get_rank()) + '_save.pt', map_location=torch.device('cpu'))
    return checkpoint



class InteractionType(Enum):
    ORIGINAL = "original"
    DCN = "dcn"
    PROJECTION = "projection"

    def __str__(self):
        return self.value


def load_snapshot(model, snapshot_dir):
    from torchsnapshot import Snapshot
    snapshot = Snapshot(path=snapshot_dir)
    snapshot.restore(app_state={"model": model})


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="torchrec dlrm example trainer")
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="number of epochs to train",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch size to use for training",
    )
    parser.add_argument(
        "--drop_last_training_batch",
        dest="drop_last_training_batch",
        action="store_true",
        help="Drop the last non-full training batch",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=None,
        help="batch size to use for validation and testing",
    )
    parser.add_argument(
        "--limit_train_batches",
        type=int,
        default=None,
        help="number of train batches",
    )
    parser.add_argument(
        "--limit_val_batches",
        type=int,
        default=None,
        help="number of validation batches",
    )
    parser.add_argument(
        "--limit_test_batches",
        type=int,
        default=None,
        help="number of test batches",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["criteo_1t", "criteo_kaggle"],
        default="criteo_1t",
        help="dataset for experiment, current support criteo_1tb, criteo_kaggle",
    )
    parser.add_argument(
        "--num_embeddings",
        type=int,
        default=100_000,
        help="max_ind_size. The number of embeddings in each embedding table. Defaults"
        " to 100_000 if num_embeddings_per_feature is not supplied.",
    )
    parser.add_argument(
        "--num_embeddings_per_feature",
        type=str,
        default=None,
        help="Comma separated max_ind_size per sparse feature. The number of embeddings"
        " in each embedding table. 26 values are expected for the Criteo dataset.",
    )
    parser.add_argument(
        "--dense_arch_layer_sizes",
        type=str,
        default="512,256,64",
        help="Comma separated layer sizes for dense arch.",
    )
    parser.add_argument(
        "--over_arch_layer_sizes",
        type=str,
        default="512,512,256,1",
        help="Comma separated layer sizes for over arch.",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=64,
        help="Size of each embedding.",
    )
    parser.add_argument(
        "--interaction_branch1_layer_sizes",
        type=str,
        default="2048,2048",
        help="Comma separated layer sizes for interaction branch1 (only on dlrm with projection).",
    )
    parser.add_argument(
        "--interaction_branch2_layer_sizes",
        type=str,
        default="2048,2048",
        help="Comma separated layer sizes for interaction branch2 (only on dlrm with projection).",
    )
    parser.add_argument(
        "--dcn_num_layers",
        type=int,
        default=3,
        help="Number of DCN layers in interaction layer (only on dlrm with DCN).",
    )
    parser.add_argument(
        "--dcn_low_rank_dim",
        type=int,
        default=512,
        help="Low rank dimension for DCN in interaction layer (only on dlrm with DCN).",
    )
    parser.add_argument(
        "--undersampling_rate",
        type=float,
        help="Desired proportion of zero-labeled samples to retain (i.e. undersampling zero-labeled rows)."
        " Ex. 0.3 indicates only 30pct of the rows with label 0 will be kept."
        " All rows with label 1 will be kept. Value should be between 0 and 1."
        " When not supplied, no undersampling occurs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--pin_memory",
        dest="pin_memory",
        action="store_true",
        help="Use pinned memory when loading data.",
    )
    parser.add_argument(
        "--mmap_mode",
        dest="mmap_mode",
        action="store_true",
        help="--mmap_mode mmaps the dataset."
        " That is, the dataset is kept on disk but is accessed as if it were in memory."
        " --mmap_mode is intended mostly for faster debugging. Use --mmap_mode to bypass"
        " preloading the dataset when preloading takes too long or when there is "
        " insufficient memory available to load the full dataset.",
    )
    parser.add_argument(
        "--in_memory_binary_criteo_path",
        type=str,
        default=None,
        help="Directory path containing the Criteo dataset npy files.",
    )
    parser.add_argument(
        "--synthetic_multi_hot_criteo_path",
        type=str,
        default=None,
        help="Directory path containing the MLPerf v2 synthetic multi-hot dataset npz files.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=15.0,
        help="Learning rate.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="Epsilon for Adagrad optimizer.",
    )
    parser.add_argument(
        "--shuffle_batches",
        dest="shuffle_batches",
        action="store_true",
        help="Shuffle each batch during training.",
    )
    parser.add_argument(
        "--shuffle_training_set",
        dest="shuffle_training_set",
        action="store_true",
        help="Shuffle the training set in memory. This will override mmap_mode",
    )
    parser.add_argument(
        "--validation_freq_within_epoch",
        type=int,
        default=None,
        help="Frequency at which validation will be run within an epoch.",
    )
    parser.set_defaults(
        pin_memory=None,
        mmap_mode=None,
        drop_last=None,
        shuffle_batches=None,
        shuffle_training_set=None,
    )
    parser.add_argument(
        "--adagrad",
        dest="adagrad",
        action="store_true",
        help="Flag to determine if adagrad optimizer should be used.",
    )
    parser.add_argument(
        "--interaction_type",
        type=InteractionType,
        choices=list(InteractionType),
        default=InteractionType.ORIGINAL,
        help="Determine the interaction type to be used (original, dcn, or projection)"
        " default is original DLRM with pairwise dot product",
    )
    parser.add_argument(
        "--collect_multi_hot_freqs_stats",
        dest="collect_multi_hot_freqs_stats",
        action="store_true",
        help="Flag to determine whether to collect stats on freq of embedding access.",
    )
    parser.add_argument(
        "--multi_hot_sizes",
        type=str,
        default=None,
        help="Comma separated multihot size per sparse feature. 26 values are expected for the Criteo dataset.",
    )
    parser.add_argument(
        "--multi_hot_distribution_type",
        type=str,
        choices=["uniform", "pareto"],
        default=None,
        help="Multi-hot distribution options.",
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--lr_decay_start", type=int, default=0)
    parser.add_argument("--lr_decay_steps", type=int, default=0)
    parser.add_argument(
        "--print_lr",
        action="store_true",
        help="Print learning rate every iteration.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Enable TensorFloat-32 mode for matrix multiplications on A100 (or newer) GPUs.",
    )
    parser.add_argument(
        "--print_sharding_plan",
        action="store_true",
        help="Print the sharding plan used for each embedding table.",
    )
    parser.add_argument("--use_xpu", action="store_true", default=False)
    parser.add_argument(
        "--log_freq",
        type=int,
        default=0,
        help="log_freq to print performance statistics.",
    )
    parser.add_argument(
        "--converge_freq",
        type=int,
        default=0,
        help="converge_freq to save model.",
    )
    parser.add_argument("--numpy_rand_seed", type=int, default=123)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--dense", action="store_true", default=False)
    parser.add_argument("--inference_only", action="store_true", default=False)
    parser.add_argument("--converge", action="store_true", default=False)
    parser.add_argument("--train_with_val", action="store_true", default=False)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--trans_data", action="store_true", default=False)
    parser.add_argument(
        "--ipex_optimize",
        action="store_true",
        help="Whether optimize model by ipex.optimize",
    )
    parser.add_argument(
        "--ipex_dist_merged_emb_adagrad",
        action="store_true",
        help="whether use ipex customer op for distributed merged embedding",
    )
    parser.add_argument(
    "--sharding_plan",
    help="Sharding plan to use",
    type=str,
    choices=["round_robin", "uniform", "auto", "hier_auto", "custom", "max_min"],
    default="round_robin",)
    parser.add_argument(
        "--snapshot_dir",
        help="Weight path for dlrm-v2",
        type=str,
    )

    parser.add_argument(
        "--checkpoint_floder",
        help="checkpoint path for dlrm-v2",
        type=str,
    )
    parser.add_argument(
        "--dynamo",
        action="store_true",
        help="Use torch compile for model test",
        default=False
    )



    return parser.parse_args(argv)


def _evaluate(
    limit_batches: Optional[int],
    #eval_pipeline: TrainPipelineSparseDist,
    eval_model,
    eval_dataloader: DataLoader,
    stage: str,
    log_freq: int,
    device: torch.device,
    args: argparse.Namespace,
) -> float:
    """
    Evaluates model. Computes and prints AUROC. Helper function for train_val_test.

    Args:
        limit_batches (Optional[int]): Limits the dataloader to the first `limit_batches` batches.
        eval_pipeline (TrainPipelineSparseDist): pipelined model.
        eval_dataloader (DataLoader): Dataloader for either the validation set or test set.
        stage (str): "val" or "test".

    Returns:
        float: auroc result
    """
    eval_model.eval()
    global n_emb, local_emb_indices, default_embedding_names, idx_name

    if args.dynamo and (not args.ipex_optimize):
        if args.amp:
            if args.use_device == "xpu":
                with torch.no_grad():
                    print(" Use dynamo for compiling")
                    if args.amp_dtype == torch.bfloat16 or args.amp_dtype == torch.float16:
                        with torch.autocast(device_type="xpu", enabled=args.amp, dtype=args.amp_dtype):
                            eval_model = torch.compile(eval_model, backend="inductor", options={"freezing": True} )
                    else:
                        eval_model = torch.compile(eval_model, backend="inductor", options={"freezing": True} )

    def eval_step(model, batch):
        batch_size = batch.dense_features.size(0)
        if args.amp:
            batch.dense_features = batch.dense_features[ext_dist.get_my_slice(batch_size)].to(device)
        else:
            batch.dense_features = batch.dense_features[ext_dist.get_my_slice(batch_size)].to(device).to(args.amp_dtype)
        batch.labels = batch.labels[ext_dist.get_my_slice(batch_size)].to(device)
        if not args.ipex_optimize:
            for k, v in batch.sparse_features.items():
                batch.sparse_features[k]['values'] = batch.sparse_features[k]['values'].to(device)
                batch.sparse_features[k]['offsets'] = batch.sparse_features[k]['offsets'].to(device)

        #if args.converge and args.dense and (args.use_device == "xpu"):
        #    torch.xpu.empty_cache()
        with torch.no_grad():
            if args.amp:
                if args.use_device == "xpu":
                    if args.amp_dtype == torch.bfloat16 or args.amp_dtype == torch.float16:
                        with torch.autocast(device_type="xpu", enabled=args.amp, dtype=args.amp_dtype):
                            logits = model(batch.dense_features, batch.sparse_features)
                    else:
                        logits = model(batch.dense_features, batch.sparse_features)
                elif args.use_device == "cuda":
                    with torch.cuda.amp.autocast(enabled=args.amp, dtype=args.amp_dtype):
                        logits = model(batch.dense_features, batch.sparse_features)
                else:
                    with torch.cpu.amp.autocast(enabled=args.amp, dtype=args.amp_dtype):
                        logits = model(batch.dense_features, batch.sparse_features)
            else:
                logits = model(batch.dense_features, batch.sparse_features)
        #if args.converge and args.dense and (args.use_device == "xpu"):
        #    torch.xpu.empty_cache()
        return logits, batch.labels
    # Set eval_pipeline._connected to False to cause the pipeline to refill with new batches as if it were newly created and empty.
    #eval_pipeline._connected = False

    iterator = itertools.islice(iter(eval_dataloader), limit_batches)

    auroc = metrics.AUROC(compute_on_step=False, task='binary').to(device)

    is_rank_zero = dist.get_rank() == 0
    total_t = 0
    it = 1
    with torch.no_grad():
        while True:
            try:
                batch = next(iterator)
                if not args.ipex_optimize:
                    batch.sparse_features = unpack(batch.sparse_features, default_embedding_names)
                #ext_dist.barrier()
                if args.use_device == "xpu":
                    torch.xpu.synchronize()
                elif args.use_device == "cuda":
                    torch.cuda.synchronize()
                else:
                    pass
                t1 = time.time()
                with (
                    contextlib.nullcontext(None) if not args.enable_profiling else
                    torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU,
                                    torch.profiler.ProfilerActivity.XPU],
                        record_shapes=False,
                    ) if args.use_xpu else
                    torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU],
                        record_shapes=False,
                    )
                ) as prof:
                    logits, labels = eval_step(eval_model, batch)
                    preds = torch.sigmoid(logits)
                    auroc(preds, labels)
                    if args.use_device == "xpu":
                        torch.xpu.synchronize()
                    elif args.use_device == "cuda":
                        torch.cuda.synchronize()
                    else:
                        pass
                    if it > 5 and args.inference_only:
                        total_t += time.time() - t1
                        logger.info(f"avg eval time per iter at ITER: {it - 5}, {total_t/(it-5)} s")
            except StopIteration:
                break
            if args.enable_profiling and it == 35 and args.inference_only:
                rank_index = ""
                if ext_dist.my_size > 1:
                    rank_index = "rank_" + str(dist.get_rank()) + "_"
                path ='./report/dlrm_inference_' + rank_index + 'profiling'
                if args.use_xpu:
                    torch.save(prof.key_averages().table(sort_by="self_xpu_time_total", row_limit=100000), \
                        path + '.pt')
                    prof.export_chrome_trace(path + '.json')
                    # Cannot sort by id when using kineto
                    # torch.save(prof.table(sort_by="id", row_limit=100000), path + '_detail.pt')
                elif args.use_device == "cuda":
                    torch.save(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=100000), \
                        path + '.pt')
                    prof.export_chrome_trace(path + '.json')
                else:
                    pass
            it += 1

    auroc_result = auroc.compute().item()
    num_samples = torch.tensor(sum(map(len, auroc.target)), device=device)
    dist.reduce(num_samples, 0, op=dist.ReduceOp.SUM)

    if is_rank_zero:
        print(f"AUROC over {stage} set: {auroc_result}.")
        print(f"Number of {stage} samples: {num_samples}")
    return auroc_result

def _train(
    train_model,
    train_optimizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    epoch: int,
    lr_scheduler,
    print_lr: bool,
    validation_freq: Optional[int],
    limit_train_batches: Optional[int],
    limit_val_batches: Optional[int],
    log_freq: int,
    device: torch.device,
    args: argparse.Namespace,
) -> bool:
    """
    Trains model for 1 epoch. Helper function for train_val_test.
    Args:
        train_pipeline (TrainPipelineSparseDist): pipelined model used for training.
        val_pipeline (TrainPipelineSparseDist): pipelined model used for validation.
        train_dataloader (DataLoader): Training set's dataloader.
        val_dataloader (DataLoader): Validation set's dataloader.
        epoch (int): The number of complete passes through the training set so far.
        lr_scheduler (LRPolicyScheduler): Learning rate scheduler.
        print_lr (bool): Whether to print the learning rate every training step.
        validation_freq (Optional[int]): The number of training steps between validation runs within an epoch.
        limit_train_batches (Optional[int]): Limits the training set to the first `limit_train_batches` batches.
        limit_val_batches (Optional[int]): Limits the validation set to the first `limit_val_batches` batches.
        print_progress (bool): Whether to print tqdm progress bar.
    Returns:
        bool: Whether the validation_auroc threshold is reached.
    """
    train_model.train()
    global n_emb, local_emb_indices, default_embedding_names, idx_name
    def train_step(model, opt, next_batch):
        batch_size = next_batch.dense_features.size(0)
        opt.zero_grad(set_to_none=True)
        if args.amp:
            next_batch.dense_features = next_batch.dense_features[ext_dist.get_my_slice(batch_size)].to(device)
        else:
            next_batch.dense_features = next_batch.dense_features[ext_dist.get_my_slice(batch_size)].to(device).to(args.amp_dtype)
        next_batch.labels = next_batch.labels[ext_dist.get_my_slice(batch_size)].to(device)

        if not args.ipex_optimize:
            for k, v in next_batch.sparse_features.items():
                next_batch.sparse_features[k]['values'] = next_batch.sparse_features[k]['values'].to(device)
                next_batch.sparse_features[k]['offsets'] = next_batch.sparse_features[k]['offsets'].to(device)
        #if args.converge and args.dense and (args.use_device == "xpu"):
        #    torch.xpu.empty_cache()
        if args.amp:
            if args.use_device == "xpu":
                if args.amp_dtype == torch.bfloat16 or args.amp_dtype == torch.float16:
                    with torch.autocast(device_type="xpu", enabled=args.amp, dtype=args.amp_dtype):
                        losses, _ = model(next_batch)
                        loss = torch.sum(losses, dim=0)
                else:
                    losses, _ = model(next_batch)
                    loss = torch.sum(losses, dim=0)
            elif args.use_device == "cuda":
                with torch.cuda.amp.autocast(enabled=args.amp, dtype=args.amp_dtype):
                    losses, _ = model(next_batch)
                    loss = torch.sum(losses, dim=0)
            else:
                with torch.cpu.amp.autocast(enabled=args.amp, dtype=args.amp_dtype):
                    losses, _ = model(next_batch)
                    loss = torch.sum(losses, dim=0)
        else:
            losses, _ = model(next_batch)
            loss = torch.sum(losses, dim=0)

        #if args.converge and args.dense and (args.use_device == "xpu"):
        #    torch.xpu.empty_cache()
        loss.backward()
        opt.step()
    start_item = 0
    if args.converge:
        start_item = args.step
    iterator = itertools.islice(iter(train_dataloader), start_item, limit_train_batches)

    is_rank_zero = dist.get_rank() == 0

    total_t = 0
    for it in itertools.count(start_item + 1, 1):
        try:
            if is_rank_zero and print_lr:
                for i, g in enumerate(train_pipeline._optimizer.param_groups):
                    print(f"lr: {it} {i} {g['lr']:.6f}")
            next_batch = next(iterator)
            if not args.ipex_optimize:
                next_batch.sparse_features = unpack(next_batch.sparse_features, default_embedding_names)
            #ext_dist.barrier()
            if args.use_device == "xpu":
                torch.xpu.synchronize()
            elif args.use_device == "cuda":
                torch.cuda.synchronize()
            else:
                pass
            t1 = time.time()
            with (
                contextlib.nullcontext(None) if not args.enable_profiling else
                torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.XPU],
                    record_shapes=False,
                ) if args.use_xpu else
                torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU],
                    record_shapes=False,
                )
            ) as prof:
                train_step(train_model, train_optimizer, next_batch)
                if args.use_device == "xpu":
                    torch.xpu.synchronize()
                elif args.use_device == "cuda":
                    torch.cuda.synchronize()
                else:
                    pass

                if it > 5:
                    total_t += time.time() - t1
                    #if log_freq != 0 and it % log_freq == 0:
                    logger.info(f"avg training time per iter at ITER: {it - 5}, {total_t/(it - 5)} s")
            if args.enable_profiling and it == 35:
                rank_index = ""
                if ext_dist.my_size > 1:
                    rank_index = "rank_" + str(dist.get_rank()) + "_"
                path ='./report/dlrm_training_' + rank_index + 'profiling'
                if args.use_device == "xpu":
                    torch.save(prof.key_averages().table(sort_by="self_xpu_time_total", row_limit=100000), \
                        path + '.pt')
                    prof.export_chrome_trace(path + '.json')
                    # Cannot sort by id when using kineto
                    # torch.save(prof.table(sort_by="id", row_limit=100000), path + '_detail.pt')
                elif args.use_device == "cuda":
                    torch.save(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=100000), \
                        path + '.pt')
                    prof.export_chrome_trace(path + '.json')
                else:
                    torch.save(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=100000), \
                        path + '.pt')
                    prof.export_chrome_trace(path + '.json')
                    pass
            lr_scheduler.step()

            if validation_freq and (it % validation_freq == 0) and args.train_with_val:
                auroc_result = _evaluate(
                    limit_val_batches,
                    train_model.model,
                    val_dataloader,
                    "val",
                    log_freq, 
                    device, 
                    args
                )
                if is_rank_zero:
                    logger.info(f"The val auc = {auroc_result}")
                train_model.train()

            if args.converge and it  % args.converge_freq == 0:
                model_state_dict = {}
                for k, v in train_model.state_dict().items():
                    k_split = k.split("module")
                    k_str = k_split[0][:-1] + k_split[1] if len(k_split) == 2 else k
                    model_state_dict[k_str] = v
                torch.save({
                    "model": model_state_dict, 
                    "optimizer": train_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "step": it}, 
                    args.checkpoint_floder + '/dlrm_training_' + str(dist.get_rank()) + '_save.pt')
        except StopIteration:
            # Dataset traversal complete
            break

    logger.info(f"Total number of iterations: {it - 1}")


@dataclass
class TrainValTestResults:
    val_aurocs: List[float] = field(default_factory=list)
    test_auroc: Optional[float] = None


def train_val_test(
    args: argparse.Namespace,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    test_dataloader: DataLoader,
    lr_scheduler: LRPolicyScheduler,
) -> TrainValTestResults:
    """
    Train/validation/test loop.

    Args:
        args (argparse.Namespace): parsed command line args.
        model (torch.nn.Module): model to train.
        optimizer (torch.optim.Optimizer): optimizer to use.
        device (torch.device): device to use.
        train_dataloader (DataLoader): Training set's dataloader.
        val_dataloader (DataLoader): Validation set's dataloader.
        test_dataloader (DataLoader): Test set's dataloader.
        lr_scheduler (LRPolicyScheduler): Learning rate scheduler.

    Returns:
        TrainValTestResults.
    """
    results = TrainValTestResults()
    if args.fp16 and (not args.amp):
        model.to(torch.float16)
    if args.bf16 and (not args.amp):
        model.to(torch.bfloat16)

    start_epoch = 0
    if args.converge:
        start_epoch = args.epoch
    if not args.inference_only:
        for epoch in range(start_epoch, args.epochs):
            _train(
                model,
                optimizer,
                train_dataloader,
                val_dataloader,
                epoch,
                lr_scheduler,
                args.print_lr,
                args.validation_freq_within_epoch,
                args.limit_train_batches,
                args.limit_val_batches,
                args.log_freq,
                device,
                args
            )
            if args.train_with_val:
                val_auroc = _evaluate(
                    args.limit_val_batches, model.model, val_dataloader, "val", args.log_freq, device, args
                )
                results.val_aurocs.append(val_auroc)
        if args.train_with_val:
            test_auroc = _evaluate(
                args.limit_test_batches, model.model, test_dataloader, "test", args.log_freq, device, args
            )

    else:
        test_auroc = _evaluate(
            args.limit_test_batches, model.model, test_dataloader, "test", args.log_freq, device, args
        )
        results.test_auroc = test_auroc

    return results


def main(argv: List[str]) -> None:
    """
    Trains, validates, and tests a Deep Learning Recommendation Model (DLRM)
    (https://arxiv.org/abs/1906.00091). The DLRM model contains both data parallel
    components (e.g. multi-layer perceptrons & interaction arch) and model parallel
    components (e.g. embedding tables). The DLRM model is pipelined so that dataloading,
    data-parallel to model-parallel comms, and forward/backward are overlapped. Can be
    run with either a random dataloader or an in-memory Criteo 1 TB click logs dataset
    (https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/).

    Args:
        argv (List[str]): command line args.

    Returns:
        None.
    """
    args = parse_args(argv)
    args.amp_dtype = torch.float32
    if args.bf16:
        args.amp_dtype = torch.bfloat16
    if args.fp16:
        args.amp_dtype = torch.float16
    for name, val in vars(args).items():
        try:
            vars(args)[name] = list(map(int, val.split(",")))
        except (ValueError, AttributeError):
            pass
    args.enable_profiling = False
    if os.getenv('PROFILE') == '1': args.enable_profiling=True

    if args.use_xpu:
        import intel_extension_for_pytorch
        try:
            import oneccl_bindings_for_pytorch
        except ImportError as e:
            pass

    if not args.use_xpu:
        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32

    if args.multi_hot_sizes is not None and args.trans_data:
        assert (
            args.num_embeddings_per_feature is not None
            and len(args.multi_hot_sizes) == len(args.num_embeddings_per_feature)
            or args.num_embeddings_per_feature is None
            and len(args.multi_hot_sizes) == len(DEFAULT_CAT_NAMES)
        ), "--multi_hot_sizes must be a comma delimited list the same size as the number of embedding tables."
    assert (
        args.in_memory_binary_criteo_path is None
        or args.synthetic_multi_hot_criteo_path is None
    ), "--in_memory_binary_criteo_path and --synthetic_multi_hot_criteo_path are mutually exclusive CLI arguments."
    assert (
        args.multi_hot_sizes is None or (not args.trans_data) or args.synthetic_multi_hot_criteo_path is None
    ), "--multi_hot_sizes is used to convert 1-hot to multi-hot. It's inapplicable with --synthetic_multi_hot_criteo_path."
    assert (
        args.multi_hot_distribution_type is None
        or args.synthetic_multi_hot_criteo_path is None
    ), "--multi_hot_distribution_type is used to convert 1-hot to multi-hot. It's inapplicable with --synthetic_multi_hot_criteo_path."

    np.random.seed(args.numpy_rand_seed)
    torch.manual_seed(args.numpy_rand_seed)
    #rank = int(os.environ["LOCAL_RANK"])
    rank = env2int(
        [
            "MPI_LOCALRANKID",
            "OMPI_COMM_WORLD_LOCAL_RANK",
            "MV2_COMM_WORLD_LOCAL_RANK",
            "LOCAL_RANK",
        ],
        0,
    )
    args.use_device = "cpu"
    if args.use_xpu and torch.xpu.is_available():
        torch.xpu.manual_seed_all(args.numpy_rand_seed)
        device: torch.device = torch.device("xpu", rank)
        backend = "ccl"
        torch.xpu.set_device(device)
        args.use_device = "xpu"
    elif torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.numpy_rand_seed)
        device: torch.device = torch.device(f"cuda:{rank}")
        backend = "nccl"
        torch.cuda.set_device(device)
        args.use_device = "cuda"
    else:
        device: torch.device = torch.device("cpu")
        backend = "gloo"

    if rank == 0:
        print(
            "PARAMS: (lr, batch_size, warmup_steps, decay_start, decay_steps): "
            f"{(args.learning_rate, args.batch_size, args.lr_warmup_steps, args.lr_decay_start, args.lr_decay_steps)}"
        )
    ext_dist.init_distributed(
        local_rank=-1, use_device=args.use_device, backend=backend
    )

    if args.num_embeddings_per_feature is not None:
        args.num_embeddings = None

    # Sets default limits for random dataloader iterations when left unspecified.
    if (
        args.in_memory_binary_criteo_path
        is args.synthetic_multi_hot_criteo_path
        is None
    ):
        for split in ["train", "val", "test"]:
            attr = f"limit_{split}_batches"
            if getattr(args, attr) is None:
                setattr(args, attr, 10)

    train_dataloader = get_dataloader(args, backend, "train")
    val_dataloader = get_dataloader(args, backend, "val")
    test_dataloader = get_dataloader(args, backend, "test")
    # use dist_models, if not you can comment out
    global n_emb, local_emb_indices, default_embedding_names, idx_name
    n_emb = len(DEFAULT_CAT_NAMES)
    #if dist.get_world_size() == 8 and dist.get_rank() == 6:
    #    local_emb_indices[1] = 25
    #if dist.get_world_size() == 8 and dist.get_rank() == 7:
    #    local_emb_indices[2] = 21

    if args.ipex_optimize:
        eb_configs = [
            EmbeddingBagConfig(
                name=f"t_{feature_name}",
                embedding_dim=args.embedding_dim,
                num_embeddings=none_throws(args.num_embeddings_per_feature)[feature_idx]
                if args.num_embeddings is None
                else args.num_embeddings,
                feature_names=[feature_name],
            )
            for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES)
        ]
    else:
        shard_matrix_str, shard_strategy_str, shard_matrix, shard_strategy = sharding.generate_plan(
        args.num_embeddings_per_feature, args.multi_hot_sizes, args.num_nodes, dist.get_world_size(), args, dist.get_rank() == 0)
        #local_emb_slice = ext_dist.get_my_slice(n_emb)
        local_emb_indices = shard_matrix[rank]
        default_embedding_names = [DEFAULT_CAT_NAMES[i] for i in local_emb_indices]
        idx_name = zip(local_emb_indices, default_embedding_names)

        eb_configs = [
            EmbeddingBagConfig(
                name=f"t_{feature_name}",
                embedding_dim=args.embedding_dim,
                num_embeddings=none_throws(args.num_embeddings_per_feature)[feature_idx]
                if args.num_embeddings is None
                else args.num_embeddings,
                feature_names=[feature_name],
            )
            #for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES)
            for feature_idx, feature_name in idx_name
        ]
    sharded_module_kwargs = {}
    if args.over_arch_layer_sizes is not None:
        sharded_module_kwargs["over_arch_layer_sizes"] = args.over_arch_layer_sizes

    if args.interaction_type == InteractionType.ORIGINAL:
        dlrm_model = DLRM(
            embedding_bag_collection=EmbeddingBagCollection(
                tables=eb_configs, device=torch.device("meta")
            ),
            dense_in_features=len(DEFAULT_INT_NAMES),
            dense_arch_layer_sizes=args.dense_arch_layer_sizes,
            over_arch_layer_sizes=args.over_arch_layer_sizes,
            dense_device=device,
        )
    elif args.interaction_type == InteractionType.DCN:
        if args.ipex_optimize:
            dlrm_model = DLRM_DCN(
                embedding_bag_collection=EmbeddingBagCollection(
                    tables=eb_configs, device=torch.device("meta")
                ),
                dense_in_features=len(DEFAULT_INT_NAMES),
                dense_arch_layer_sizes=args.dense_arch_layer_sizes,
                over_arch_layer_sizes=args.over_arch_layer_sizes,
                dcn_num_layers=args.dcn_num_layers,
                dcn_low_rank_dim=args.dcn_low_rank_dim,
                dense_device=device,
            )
        else:
            dlrm_model = DIST_DLRM_DCN(
                embedding_bag_collection=EmbeddingBagCollection(
                    tables=eb_configs, device="cpu"
                ),
                dense_in_features=len(DEFAULT_INT_NAMES),
                num_total_sparse_features=n_emb,
                dense_arch_layer_sizes=args.dense_arch_layer_sizes,
                over_arch_layer_sizes=args.over_arch_layer_sizes,
                dcn_num_layers=args.dcn_num_layers,
                dcn_low_rank_dim=args.dcn_low_rank_dim,
                n_emb=n_emb,
                shard_matrix=shard_matrix, 
                shard_strategy = shard_strategy,
                args=args,
                dense_device="cpu",
            )
            for embedding_bag in dlrm_model.sparse_arch.embedding_bag_collection.embedding_bags.values():
                _W = np.random.uniform(
                    low=-np.sqrt(1 / embedding_bag.num_embeddings),
                    high=np.sqrt(1 / embedding_bag.num_embeddings),
                    size=(embedding_bag.num_embeddings, embedding_bag.embedding_dim)).astype(np.float32)
                embedding_bag.weight.data = torch.tensor(_W, requires_grad=True)

    elif args.interaction_type == InteractionType.PROJECTION:
        dlrm_model = DLRM_Projection(
            embedding_bag_collection=EmbeddingBagCollection(
                tables=eb_configs, device=torch.device("meta")
            ),
            dense_in_features=len(DEFAULT_INT_NAMES),
            dense_arch_layer_sizes=args.dense_arch_layer_sizes,
            over_arch_layer_sizes=args.over_arch_layer_sizes,
            interaction_branch1_layer_sizes=args.interaction_branch1_layer_sizes,
            interaction_branch2_layer_sizes=args.interaction_branch2_layer_sizes,
            dense_device=device,
        )
    else:
        raise ValueError(
            "Unknown interaction option set. Should be original, dcn, or projection."
        )

    if not args.amp:
        dlrm_model.to(args.amp_dtype)
    train_model = DLRMTrain(dlrm_model)
    if args.inference_only:
        assert args.snapshot_dir
        load_snapshot(train_model, args.snapshot_dir)


    # This will apply the Adagrad optimizer in the backward pass for the embeddings (sparse_arch). This means that
    # the optimizer update will be applied in the backward pass, in this case through a fused op.
    # TorchRec will use the FBGEMM implementation of EXACT_ADAGRAD. For GPU devices, a fused CUDA kernel is invoked. For CPU, FBGEMM_GPU invokes CPU kernels
    # https://github.com/pytorch/FBGEMM/blob/2cb8b0dff3e67f9a009c4299defbd6b99cc12b8f/fbgemm_gpu/fbgemm_gpu/split_table_batched_embeddings_ops.py#L676-L678

    # Note that lr_decay, weight_decay and initial_accumulator_value for Adagrad optimizer in FBGEMM v0.3.2
    # cannot be specified below. This equivalently means that all these parameters are hardcoded to zero.
    optimizer_kwargs = {"lr": args.learning_rate}
    if args.adagrad:
        optimizer_kwargs["eps"] = args.eps

    model = train_model
    optimizer_param = None
    if args.ipex_optimize:
        model.model, optimizer_param = replace_embeddingbag_collection(model.model, device, args)
    else:
        model.model.sparse_arch = SparseArchTraceAbleWrapper(model.model.sparse_arch, args.dense)
        optimizer_param = model.parameters()
    print(model)
    checkpoint = None
    args.epoch = 0
    args.step = 0
    if args.converge:
        checkpoint = load_checkoutpoint(args, device)
        if checkpoint is not None:
            args.epoch = checkpoint['epoch']
            args.step = checkpoint['step']
            print('[info] loadind model state dict...')
            train_model.load_state_dict(checkpoint['model'])
        logger.info(f"The start epoch is : {args.epoch}")
        logger.info(f"The start step is : {args.step}")

    optimizer = None
    if not args.inference_only:
        if args.adagrad:
            optimizer = torch.optim.Adagrad(optimizer_param, lr=args.learning_rate, eps=args.eps)
        else:
            optimizer = torch.optim.SGD(optimizer_param, lr=args.learning_rate)
        if args.converge and checkpoint is not None:
            print('[info] loadind optimizer state dict...')
            optimizer.load_state_dict(checkpoint['optimizer'])

    if ext_dist.my_size > 1:
        if args.use_device == "cuda":
            device_ids = [rank]
            model.to(device)
            model.model.dense_arch = ext_dist.DDP(model.model.dense_arch, device_ids=device_ids)
            model.model.inter_arch = ext_dist.DDP(model.model.inter_arch, device_ids=device_ids)
            model.model.over_arch = ext_dist.DDP(model.model.over_arch, device_ids=device_ids)
            if not args.inference_only:
                for k, v in optimizer.state.items():
                    for name, value in v.items():
                        if isinstance(value, torch.Tensor):
                            optimizer.state[k][name] = value.to(device)
        elif args.use_device == "xpu":
            device_ids = [device]
            model.to(device)
            model.model.dense_arch = ext_dist.DDP(model.model.dense_arch, device_ids=device_ids)
            model.model.inter_arch = ext_dist.DDP(model.model.inter_arch, device_ids=device_ids)
            model.model.over_arch = ext_dist.DDP(model.model.over_arch, device_ids=device_ids)
            if (not args.inference_only):
                for k, v in optimizer.state.items():
                    for name, value in v.items():
                        if isinstance(value, torch.Tensor):
                            optimizer.state[k][name] = value.to(device)
        else:
            model.model.dense_arch = ext_dist.DDP(model.model.dense_arch)
            model.model.inter_arch = ext_dist.DDP(model.model.inter_arch)
            model.model.over_arch = ext_dist.DDP(model.model.over_arch)

    lr_scheduler = None
    if not args.inference_only:
        lr_scheduler = LRPolicyScheduler(
            optimizer, args.lr_warmup_steps, args.lr_decay_start, args.lr_decay_steps
        )
        if args.converge and checkpoint is not None:
            print('[info] loadind lr_scheduler state dict...')
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        model.train()
        if args.use_device == "xpu":
            model, optimizer = torch.xpu.optimize(model=model, optimizer=optimizer, level="O1",
                                                  dtype=torch.float32)
    else:
        model.eval()
        if args.use_device == "xpu":
            model = torch.xpu.optimize(model=model, dtype=args.amp_dtype)

    if args.trans_data and (args.multi_hot_sizes is not None):
        multihot = Multihot(
            args.multi_hot_sizes,
            args.num_embeddings_per_feature,
            args.batch_size,
            collect_freqs_stats=args.collect_multi_hot_freqs_stats,
            dist_type=args.multi_hot_distribution_type,
        )
        multihot.pause_stats_collection_during_val_and_test(model)
        train_dataloader = RestartableMap(
            multihot.convert_to_multi_hot, train_dataloader
        )
        val_dataloader = RestartableMap(multihot.convert_to_multi_hot, val_dataloader)
        test_dataloader = RestartableMap(multihot.convert_to_multi_hot, test_dataloader)
    train_val_test(
        args,
        model,
        optimizer,
        device,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        lr_scheduler,
    )
    if args.collect_multi_hot_freqs_stats:
        multihot.save_freqs_stats()


if __name__ == "__main__":
    main(sys.argv[1:])
