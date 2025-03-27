#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=masked-lm
"""
# You can also adapt this script on your own mlm task. Pointers for this are left as comments.

"""BERT Pretraining"""
import argparse
import csv
import h5py
import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
import logging
import math
import multiprocessing
import random
import re
import time
import sys

from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor

from schedulers import LinearWarmupPolyDecayScheduler

import utils

import torch.nn.functional as F
import argparse

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForPreTraining,
    SchedulerType,
    get_scheduler,
    set_seed,
)

from schedulers import LinearWarmUpScheduler, LinearWarmupPolyDecayScheduler
from lamb import Lamb

try:
    import intel_extension_for_pytorch as ipex
    from intel_extension_for_pytorch.quantization.fp8 import (
        fp8_autocast,
        DelayedScaling,
        Format,
        prepare_fp8,
    )
except:
    ipex = None

ref_time = 0
PRINT_ITER_FACTOR = 32


def get_time():
    global ref_time
    t = time.time()
    return (t - ref_time) * 1000.0


def sync_clock():
    global ref_time
    t = time.time()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        t = time.time()
        t = torch.tensor([t], dtype=torch.double)
        torch.distributed.broadcast(t, 0)
        torch.distributed.barrier()
        t = t.item()
    ref_time = t


def synchronize(device="cpu"):
    if device == "cuda":
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(torch.cuda.FloatTensor(1))
            torch.cuda.synchronize()
        else:
            torch.cuda.synchronize()
    elif device == "xpu":
        torch.xpu.synchronize()


def get_time_sync(device="cpu"):
    synchronize(device)
    return get_time()


logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


class WorkerInitObj(object):
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, id):
        np.random.seed(seed=self.seed + id)
        random.seed(self.seed + id)


def get_eval_batchsize_per_worker(args):
    if torch.distributed.is_initialized():
        chunk_size = args.num_eval_examples // args.world_size
        rank = args.rank
        remainder = args.num_eval_examples % args.world_size
        if rank < remainder:
            return chunk_size + 1
        else:
            return chunk_size


class pretraining_dataset_v1(Dataset):
    def __init__(self, f, input_file, max_pred_length):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        keys = [
            "input_ids",
            "input_mask",
            "segment_ids",
            "masked_lm_positions",
            "masked_lm_ids",
            "next_sentence_labels",
        ]
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        print(f"Loaded {len(self.inputs[0]):d} samples from datafile: {input_file}")
        f.close()

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.inputs[0])

    def __getitem__(self, index):
        [
            input_ids,
            input_mask,
            segment_ids,
            masked_lm_positions,
            masked_lm_ids,
            next_sentence_labels,
        ] = [
            (
                torch.from_numpy(input[index].astype(np.int64))
                if indice < 5
                else torch.from_numpy(np.asarray(input[index].astype(np.int64)))
            )
            for indice, input in enumerate(self.inputs)
        ]
        masked_lm_labels = torch.zeros(input_ids.shape, dtype=torch.long) - 100
        index = self.max_pred_length
        masked_token_count = torch.count_nonzero(masked_lm_positions)
        if masked_token_count != 0:
            index = masked_token_count
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]
        # print(f"input_mask_len = {torch.count_nonzero(input_ids)}  index = {index}")

        return [
            input_ids,
            segment_ids,
            input_mask,
            masked_lm_labels,
            next_sentence_labels,
        ]


class pretraining_dataset_v2(Dataset):
    def __init__(
        self, f, input_file, max_pred_length, max_seq_length=512, packed_samples=False
    ):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        self.max_seq_length = max_seq_length
        self.packed_samples = packed_samples

        if not self.packed_samples:
            keys = [
                "input_ids",
                "segment_ids",
                "masked_lm_positions",
                "masked_lm_ids",
                "next_sentence_labels",
            ]
        else:
            keys = [
                "input_ids",
                "segment_ids",
                "masked_lm_positions",
                "masked_lm_ids",
                "packed_input_len",
                "packed_masked_lm_len",
                "next_sentence_labels",
            ]

        self.inputs = [np.asarray(f[key][:]) for key in keys]
        print(f"Loaded {len(self.inputs[0]):d} samples from datafile: {input_file}")
        f.close()

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.inputs[0])

    def __getitem__(self, index):
        input_ids = np.zeros((self.max_seq_length)).astype(np.int64)
        input_mask = np.zeros((self.max_seq_length)).astype(np.int64)
        segment_ids = np.zeros((self.max_seq_length)).astype(np.int64)
        next_sentence_labels = np.zeros((3)).astype(np.int64)
        packed_input_len = np.zeros((3)).astype(np.int64)

        if not self.packed_samples:
            [
                _input_ids,
                _segment_ids,
                _masked_lm_positions,
                _masked_lm_ids,
                _next_sentence_labels,
            ] = [
                (
                    input[index].astype(np.int64)
                    if indice < 4
                    else np.asarray(input[index].astype(np.int64))
                )
                for indice, input in enumerate(self.inputs)
            ]
        else:
            [
                _input_ids,
                _segment_ids,
                _masked_lm_positions,
                _masked_lm_ids,
                _packed_input_len,
                _packed_masked_lm_len,
                _next_sentence_labels,
            ] = [
                input[index].astype(np.int64)
                for indice, input in enumerate(self.inputs)
            ]

        input_mask_len = _input_ids.shape[-1]
        input_ids[:input_mask_len] = _input_ids
        input_mask[:input_mask_len] = np.ones((1, input_mask_len)).astype(np.int64)
        segment_ids[:input_mask_len] = _segment_ids
        masked_lm_labels = np.zeros(input_ids.shape, dtype=np.int64) - 100
        masked_lm_labels[_masked_lm_positions] = _masked_lm_ids

        if not self.packed_samples:
            next_sentence_labels = _next_sentence_labels

            return [
                torch.from_numpy(input_ids),
                torch.from_numpy(segment_ids),
                torch.from_numpy(input_mask),
                torch.from_numpy(masked_lm_labels),
                torch.from_numpy(next_sentence_labels),
            ]
        else:
            packed_seqs = _packed_input_len.shape[-1]
            next_sentence_labels[:packed_seqs] = _next_sentence_labels
            packed_input_len[:packed_seqs] = _packed_input_len

            return [
                torch.from_numpy(input_ids),
                torch.from_numpy(segment_ids),
                torch.from_numpy(input_mask),
                torch.from_numpy(masked_lm_labels),
                torch.from_numpy(next_sentence_labels),
                torch.from_numpy(packed_input_len),
            ]


def pretraining_dataset(
    input_file, max_pred_length, max_seq_length=512, packed_samples=False
):
    f = h5py.File(input_file, "r")
    if "input_mask" not in f.keys():
        return pretraining_dataset_v2(
            f, input_file, max_pred_length, max_seq_length, packed_samples
        )
    else:
        return pretraining_dataset_v1(f, input_file, max_pred_length)


def create_pretraining_dataset(
    input_file, max_pred_length, shared_list, args, worker_init_fn
):
    train_data = pretraining_dataset(
        input_file=input_file, max_pred_length=max_pred_length
    )
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=args.train_batch_size
    )
    return train_dataloader, input_file


def create_eval_dataset(args, worker_init_fn):
    eval_data = []
    for eval_file in sorted(os.listdir(args.eval_dir)):
        eval_file_path = os.path.join(args.eval_dir, eval_file)

        if os.path.isfile(eval_file_path) and "part" in eval_file_path:
            eval_data.extend(
                pretraining_dataset(
                    eval_file_path, max_pred_length=args.max_predictions_per_seq
                )
            )
            if len(eval_data) > args.num_eval_examples:
                eval_data = eval_data[: args.num_eval_examples]
                break
    if torch.distributed.is_initialized():
        chunk_size = args.num_eval_examples // args.world_size
        rank = args.rank
        remainder = args.num_eval_examples % args.world_size
        if rank < remainder:
            eval_data = eval_data[
                (chunk_size + 1) * rank : (chunk_size + 1) * (rank + 1)
            ]
        else:
            eval_data = eval_data[
                chunk_size * rank + remainder : chunk_size * (rank + 1) + remainder
            ]

    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(
        eval_data,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        num_workers=0,
        pin_memory=True,
    )
    return eval_dataloader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a Masked Language Modeling task"
    )

    # Required parameters
    parser.add_argument(
        "--input_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain .hdf5 files  for the task.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints will be written.",
    )
    parser.add_argument(
        "--eval_dir",
        default=None,
        type=str,
        help="The eval data dir. Should contain .hdf5 files  for the task.",
    )
    parser.add_argument(
        "--eval_iter_start_samples",
        default=3000000,
        type=int,
        help="Sample to begin performing eval.",
    )
    parser.add_argument(
        "--eval_iter_samples",
        default=16,
        type=int,
        help="If set to -1, disable eval, \
                        else evaluate every eval_iter_samples during training",
    )
    parser.add_argument(
        "--num_eval_examples",
        default=10000,
        type=int,
        help="number of eval examples to run eval on",
    )
    parser.add_argument(
        "--resume_checkpoint",
        default=None,
        type=str,
        help="Use to load checkpoint to resume training",
    )
    parser.add_argument(
        "--init_tf_checkpoint",
        default=None,
        type=str,
        help="The initial TF checkpoint to start training from.",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="A csv or a json file containing the training data.",
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="A csv or a json file containing the validation data.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--min_learning_rate",
        type=float,
        default=0.0,
        help="Minimal learning rate to use.",
    )
    parser.add_argument(
        "--max_predictions_per_seq",
        default=76,
        type=int,
        help="The maximum total of masked tokens in input sequence",
    )
    parser.add_argument(
        "--train_batch_size", default=8, type=int, help="Total batch size for training."
    )
    parser.add_argument(
        "--eval_batch_size", default=16, type=int, help="Total batch size for training."
    )
    parser.add_argument(
        "--weight_decay_rate",
        default=0.01,
        type=float,
        help="weight decay rate for LAMB.",
    )
    parser.add_argument(
        "--opt_lamb_beta_1", default=0.9, type=float, help="LAMB beta1."
    )
    parser.add_argument(
        "--opt_lamb_beta_2", default=0.999, type=float, help="LAMB beta2."
    )
    parser.add_argument(
        "--max_steps",
        default=218880,
        type=int,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--max_steps_for_scheduler",
        default=218880,
        type=int,
        help="Total number of training steps for scheduler to decrease lr.",
    )
    parser.add_argument(
        "--max_samples_termination",
        default=14000000,
        type=float,
        help="Total number of training samples to run.",
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.01,
        type=float,
        help="Proportion of optimizer update steps to perform linear learning rate warmup for. "
        "Typically 1/8th of steps for Phase2",
    )
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=float,
        help="Number of optimizer update steps to perform linear learning rate warmup for. "
        "Typically 1/8th of steps for Phase2",
    )
    parser.add_argument(
        "--start_warmup_step", default=0, type=float, help="Starting step for warmup. "
    )
    parser.add_argument(
        "--log_freq",
        type=float,
        default=10000.0,
        help="frequency of logging loss. If not positive, no logging is provided for training loss",
    )
    parser.add_argument(
        "--checkpoint_activations",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=False,
        action="store_true",
        help="Whether to resume training from checkpoint. If set, precedes init_tf_checkpoint",
    )
    parser.add_argument(
        "--keep_n_most_recent_checkpoints",
        type=int,
        default=20,
        help="Number of checkpoints to keep (rolling basis).",
    )
    parser.add_argument(
        "--num_samples_per_checkpoint",
        type=int,
        default=500000,
        help="Number of update steps until a model checkpoint is saved to disk.",
    )
    parser.add_argument(
        "--min_samples_to_start_checkpoints",
        type=int,
        default=3000000,
        help="Number of update steps until model checkpoints start saving to disk.",
    )
    parser.add_argument(
        "--skip-checkpoint",
        default=False,
        action="store_true",
        help="Whether to save checkpoints",
    )
    parser.add_argument(
        "--phase2",
        default=False,
        action="store_true",
        help="Only required for checkpoint saving format",
    )
    parser.add_argument(
        "--do_train",
        default=False,
        action="store_true",
        help="Whether to run training.",
    )
    parser.add_argument(
        "--bert_config_path",
        type=str,
        default="/workspace/phase1",
        help="Path bert_config.json is located in",
    )
    parser.add_argument(
        "--target_mlm_accuracy",
        type=float,
        default=0.72,
        help="Stop training after reaching this Masked-LM accuracy",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Stop training after reaching this Masked-LM accuracy",
    )
    parser.add_argument(
        "--train_mlm_accuracy_window_size",
        type=int,
        default=0,
        help="Average accuracy over this amount of batches before performing a stopping criterion test",
    )
    parser.add_argument(
        "--num_epochs_to_generate_seeds_for",
        type=int,
        default=2,
        help="Number of epochs to plan seeds for. Same set across all workers.",
    )
    parser.add_argument(
        "--use_gradient_as_bucket_view",
        default=False,
        action="store_true",
        help="Turn ON gradient_as_bucket_view optimization in native DDP.",
    )
    parser.add_argument(
        "--dense_seq_output",
        default=False,
        action="store_true",
        help="Whether to run with optimizations.",
    )
    parser.add_argument(
        "--no_ddp", default=False, action="store_true", help="Whether to use DDP."
    )
    parser.add_argument(
        "--converge",
        default=False,
        action="store_true",
        help="Whether to run convergence.",
    )
    parser.add_argument(
        "--dist_lamb",
        default=False,
        action="store_true",
        help="Whether to use DistLamb from pcl_bert",
    )

    # For dtype specific training
    parser.add_argument(
        "--bf16", default=False, action="store_true", help="Enale BFloat16 training"
    )
    parser.add_argument(
        "--fp8", default=False, action="store_true", help="Enale FP8 training"
    )
    parser.add_argument(
        "--bf32", default=False, action="store_true", help="Enale BFloat32 training"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )

    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--rank", default=0, type=int, help="node rank for distributed training"
    )
    parser.add_argument(
        "--world_size", default=1, type=int, help="Total batch size for training."
    )

    parser.add_argument(
        "--profile", action="store_true", help="Whether to enable profiling"
    )
    parser.add_argument(
        "--export_chrome_trace",
        action="store_true",
        help="Exports the collected trace in Chrome JSON format.",
    )
    parser.add_argument(
        "--skip_exchange_padding",
        default=False,
        action="store_true",
        help="Whether to skip padding exchange.",
    )
    parser.add_argument(
        "--unpad",
        default=False,
        action="store_true",
        help="Whether to run with unpadding.",
    )
    parser.add_argument(
        "--benchmark_steps",
        type=int,
        default=0,
        help="Number of steps to run for benchmark.",
    )
    parser.add_argument(
        "--dist_backend",
        type=str,
        default="ccl",
        help="Specify distributed backend to use.",
    )
    parser.add_argument(
        "--dist-url",
        default="127.0.0.1",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-port",
        default="29500",
        type=str,
        help="url port used to set up distributed training",
    )
    parser.add_argument(
        "--large-first-bucket",
        action="store_true",
        help="Configure a large capacity of the first bucket in DDP for allreduce",
    )
    parser.add_argument(
        "--disable-broadcast-buffers",
        action="store_true",
        help="disable syncing buffers",
    )
    parser.add_argument(
        "--multi_instance",
        action="store_true",
        help="Skip torch.distributed initialization to Run multiple instance independently",
    )
    parser.add_argument(
        "--dist_profile",
        action="store_true",
        help="Whether to enable distributed timing profile",
    )
    parser.add_argument(
        "--bucket_cap",
        default=25,
        type=int,
        help="controls the bucket size in MegaBytes",
    )

    # choose device
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="backend to run",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Whether to enable autocast",
    )
    parser.add_argument("--lamb", action="store_true")
    parser.add_argument("--adamw", action="store_true")
    parser.add_argument("--num-iterations", default="10000000000", type=str)
    parser.add_argument("--info", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--sdp", action="store_true")
    parser.add_argument(
        "--kineto_profile",
        action="store_true",
        help="Whether to running kineto profiler",
    )

    args = parser.parse_args()
    # profiling, support both 2 methods to enable profiler
    if args.kineto_profile:
        os.environ["PROFILE"] = "1"
    if os.environ.get("PROFILE", "OFF").upper() in ["1", "Y", "ON", "YES", "TRUE"]:
        args.profile = True
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    return args


def found_resume_checkpoint(args):
    if args.phase2:
        checkpoint_str = "phase2_ckpt*.pt"
    else:
        checkpoint_str = "phase1_ckpt*.pt"
    num_ckpts = len(glob.glob(os.path.join(args.output_dir, checkpoint_str)))
    print("\n\nfound num checkpoints: ", num_ckpts)
    print("resume from checkpoints: ", args.resume_from_checkpoint)
    print("resume checkpoint: ", args.resume_checkpoint)
    if args.resume_from_checkpoint and not os.path.exists(args.resume_checkpoint):
        raise RuntimeError(
            "The checkpoint [", args.resume_checkpoint, "] doesn't exist"
        )
    return args.resume_from_checkpoint and num_ckpts > 0


def setup_training(args):
    device = torch.device(args.device)
    if (
        int(os.environ.get("PMI_SIZE", "0")) > 1 or args.world_size > 1
    ) and not args.multi_instance:
        if args.dist_backend == "ccl":
            try:
                import oneccl_bindings_for_pytorch
            except:
                print(
                    "CCL backend requested but import oneccl_bindings_for_pytorch failed"
                )
                raise
        elif args.dist_backend == "mpi":
            if not torch.distributed.is_mpi_available():
                try:
                    import torch_mpi
                except:
                    print(
                        "MPI backend requested but not available try installing torch_mpi module"
                    )
                    raise
        else:
            raise ValueError(f"{args.dist_backend} backend requested but not supported")

        if args.world_size == 1:
            mpi_world_size = int(os.environ.get("PMI_SIZE", -1))
            if mpi_world_size > 0:
                os.environ["MASTER_ADDR"] = args.dist_url  # '127.0.0.1'
                os.environ["MASTER_PORT"] = args.dist_port  # '29500'
                os.environ["RANK"] = os.environ.get("PMI_RANK", -1)
                os.environ["WORLD_SIZE"] = os.environ.get("PMI_SIZE", -1)
                args.rank = int(os.environ.get("PMI_RANK", -1))
                args.world_size = int(os.environ.get("WORLD_SIZE", -1))
        else:
            if "PMIX_RANK" in os.environ.keys():  # mpich set
                os.environ["MASTER_ADDR"] = args.dist_url  # '127.0.0.1'
                os.environ["MASTER_PORT"] = args.dist_port  # '29500'
                os.environ["RANK"] = os.environ.get("PMIX_RANK")
                os.environ["WORLD_SIZE"] = str(args.world_size)
                args.rank = int(os.environ.get("PMIX_RANK", -1))

        if args.world_size > 1:
            init_method = "tcp://" + args.dist_url + ":" + args.dist_port
            torch.distributed.init_process_group(
                backend=args.dist_backend,
                init_method=init_method,
                world_size=args.world_size,
                rank=args.rank,
            )
            # refer to https://www.open-mpi.org/faq/?category=running#mpi-environmental-variables for MPI ENV
            if "MPI_LOCALRANKID" in os.environ.keys():
                local_rank = os.environ["MPI_LOCALRANKID"]
            elif "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ.keys():
                local_rank = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
            else:  # mpich set
                local_rank = os.environ["PALS_LOCAL_RANKID"]
            device = torch.device("xpu", int(local_rank))
            print(
                "world_size:{}, rank:{}, device:{}".format(
                    args.world_size, args.rank, device
                )
            )

        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()
        if args.rank == 0:
            print(
                f"Using {args.dist_backend.upper()} dist run with {args.world_size} ranks",
                flush=True,
            )
    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps
            )
        )
    if args.train_batch_size % args.gradient_accumulation_steps != 0:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, batch size {} should be divisible".format(
                args.gradient_accumulation_steps, args.train_batch_size
            )
        )

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    if not (args.do_train or (args.eval_dir and args.eval_iter_samples <= 0)):
        raise ValueError(" `do_train`  or should be in offline eval mode")

    if not args.resume_from_checkpoint or not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    return device, args


optimize_dtype = None


def prepare_model_and_optimizer(args, device):
    global optimize_dtype
    if args.bf16:
        optimize_dtype = torch.bfloat16
    else:
        optimize_dtype = torch.float32

    global_step = 0
    args.resume_step = 0
    checkpoint = None
    # download model & vocab.
    if args.config_name:
        print("Get config from config_name", args.config_name)
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        print("Get config from model_name_or_path: ", args.model_name_or_path)
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    config.dense_seq_output = args.dense_seq_output
    if args.model_name_or_path:
        print("Load model by AutoModelForPreTraining.from_pretrained")
        model = AutoModelForPreTraining.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
        model.train()
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForPreTraining.from_config(config)
        model.train()

    if args.resume_from_checkpoint and args.resume_checkpoint is not None:
        if not os.path.exists(args.resume_checkpoint):
            raise RuntimeError(
                "The checkpoint ", args.resume_checkpoint, " doesn't exist"
            )
        print(
            "load...",
            args.resume_checkpoint,
            " to ",
            (args.device + ":" + str(args.rank)),
        )
        map_location = str(args.device) + ":" + str(args.rank)
        checkpoint = torch.load(args.resume_checkpoint, map_location=map_location)

    print("Set different weight_decay for model parameters")
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "gamma", "beta", "LayerNorm"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay_rate,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    print(
        "GroupSizes: ",
        [sum([p.numel() for p in g["params"]]) for g in optimizer_grouped_parameters],
    )

    print("Model to device: ", device)
    model = model.to(device).train()
    if torch.distributed.is_initialized() and not args.no_ddp:
        torch.xpu.set_device(device)
        model.xpu(device)
    if args.dist_lamb:
        raise NotImplementedError("Not implemeted for dist_lamb")
    elif args.adamw:
        print("using adamw")
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate
        )
    elif args.lamb:
        print("using lamb")
        optimizer = Lamb(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            betas=(args.opt_lamb_beta_1, args.opt_lamb_beta_2),
        )
    else:
        raise NotImplementedError("Please use valid optimizer")

    if args.sdp:
        model, optimizer = ipex.optimize_transformers(
            model, optimizer, dtype=optimize_dtype, device=args.device, inplace=True
        )

    if args.device == "xpu":
        print("Doing torch xpu optimize, dtype: ", optimize_dtype)
        model, optimizer = torch.xpu.optimize(
            model=model, optimizer=optimizer, dtype=optimize_dtype
        )

    if args.warmup_steps == 0:
        warmup_steps = int(args.max_steps_for_scheduler * args.warmup_proportion)
        warmup_start = 0
    else:
        warmup_steps = args.warmup_steps
        warmup_start = args.start_warmup_step

    if args.converge:
        print("LinearWarmupPolyDecayScheduler:")
        print("warmup start = ", warmup_start)
        print("warmup steps = ", warmup_steps)
        print("max steps for scheduler = ", args.max_steps_for_scheduler)
        print("initial learning rate = ", args.learning_rate)
        print("minimal learning rate = ", args.min_learning_rate)
    lr_scheduler = LinearWarmupPolyDecayScheduler(
        optimizer,
        start_warmup_steps=warmup_start,
        warmup_steps=warmup_steps,
        total_steps=args.max_steps_for_scheduler,
        end_learning_rate=args.min_learning_rate,
        degree=1.0,
    )

    # load checkpoint for model, optimizer and scheduler
    if checkpoint is not None:
        print("[info] loadind model state dict...")
        model.load_state_dict(checkpoint["model"])
        print("[info] loadind optimizer state dict...")
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("[info] loadind lr_scheduler state dict...")
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    # DDP
    if torch.distributed.is_initialized() and not args.no_ddp:
        if args.large_first_bucket:
            # set the first bucket with maximal size to cover all parameters for allreduce
            torch.distributed._DEFAULT_FIRST_BUCKET_BYTES = sys.maxsize
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device],
            find_unused_parameters=True,
            bucket_cap_mb=args.bucket_cap,
            broadcast_buffers=False if args.disable_broadcast_buffers else True,
            gradient_as_bucket_view=args.use_gradient_as_bucket_view,
        )

    if args.compile:
        model = torch.compile(model, mode="reduce-overhead")

    return model, optimizer, lr_scheduler, checkpoint, global_step


def run_eval(
    model,
    eval_dataloader,
    device,
    num_eval_examples,
    args,
    first_eval=False,
    use_cache=False,
):
    model.eval()
    total_eval_loss, total_eval_mlm_acc = 0.0, 0.0
    total_masked = 0
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            (
                input_ids,
                segment_ids,
                input_mask,
                masked_lm_labels,
                next_sentence_labels,
            ) = batch
            input_ids = input_ids.to(device)
            segment_ids = segment_ids.to(device)
            input_mask = input_mask.to(device)
            masked_lm_labels = masked_lm_labels.to(device)
            next_sentence_labels = next_sentence_labels.to(device)
            with torch.autocast("xpu", enabled=True, dtype=optimize_dtype):
                outputs = model(
                    input_ids=input_ids,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                    labels=masked_lm_labels,
                    next_sentence_label=next_sentence_labels,
                )
            mlm_acc, num_masked = calc_mlm_acc(
                outputs, masked_lm_labels, args.dense_seq_output
            )
            total_eval_loss += outputs.loss.item() * num_masked
            total_eval_mlm_acc += mlm_acc * num_masked
            total_masked += num_masked

            torch.xpu.synchronize(args.device)
            if args.converge:
                print(
                    "Eval [",
                    step,
                    "/",
                    len(eval_dataloader),
                    "] loss: ",
                    round(float(total_eval_loss / total_masked), 4),
                    " acc: ",
                    mlm_acc.item(),
                    "(avg acc: ",
                    round(float(total_eval_mlm_acc.item() / total_masked), 4),
                    " num masked: ",
                    num_masked,
                    ")",
                )

    model.train()
    total_masked = torch.tensor(total_masked, device=device, dtype=torch.int64)
    total_eval_loss = torch.tensor(total_eval_loss, device=device, dtype=torch.float64)
    if torch.distributed.is_initialized():
        # Collect total scores from all ranks
        torch.distributed.all_reduce(
            total_eval_mlm_acc, op=torch.distributed.ReduceOp.SUM
        )
        torch.distributed.all_reduce(total_eval_loss, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_masked, op=torch.distributed.ReduceOp.SUM)
        print("finish all reduce for validation loss and accuracy")

    # Average by number of examples
    total_eval_mlm_acc /= total_masked
    total_eval_loss /= total_masked

    return total_eval_loss, total_eval_mlm_acc


def global_batch_size(args):
    return args.train_batch_size * args.gradient_accumulation_steps * args.world_size


def calc_mlm_acc(outputs, masked_lm_labels, dense_seq_output=False):
    prediction_scores = outputs.prediction_logits
    masked_lm_labels_flat = masked_lm_labels.view(-1)
    mlm_labels = masked_lm_labels_flat[masked_lm_labels_flat != -100]
    if not dense_seq_output:
        prediction_scores_flat = prediction_scores.view(-1, prediction_scores.shape[-1])
        mlm_predictions_scores = prediction_scores_flat[masked_lm_labels_flat != -100]
        mlm_predictions = mlm_predictions_scores.argmax(dim=-1)
    else:
        mlm_predictions = prediction_scores.argmax(dim=-1)

    num_masked = mlm_labels.numel()
    mlm_acc = (mlm_predictions == mlm_labels).sum(dtype=torch.float) / num_masked

    return mlm_acc, num_masked


def calc_accuracy(outputs, masked_lm_labels, next_sentence_label, args):
    loss = outputs.loss.item()
    prediction_logits = outputs.prediction_logits
    seq_relationship_logits = outputs.seq_relationship_logits
    mlm_acc, num_masked = calc_mlm_acc(outputs, masked_lm_labels, args.dense_seq_output)
    seq_acc_t = (
        torch.argmax(seq_relationship_logits, dim=-1)
        .eq(next_sentence_label.view([-1]))
        .to(torch.float)
    )
    seq_acc_true, seq_tot = seq_acc_t.sum().item(), seq_acc_t.numel()
    seq_acc = seq_acc_true / seq_tot
    return loss, mlm_acc, num_masked, seq_acc, seq_tot


def main():
    global_start_time = time.time()
    latency_list = []
    args = parse_args()

    # To ensure the same model master weight at initial state, set seed before preparing the model.
    if args.seed is not None:
        print("Setting seed to ensure same model master weight at the beginning.")
        torch.manual_seed(args.seed)
        torch.xpu.manual_seed(args.seed)

    if args.converge:
        print(
            "For convergence verbose show freq, PRINT_ITER_FACTOR = ", PRINT_ITER_FACTOR
        )
    status = "aborted"  # later set to 'success' if termination criteria met
    device, args = setup_training(args)
    print("args.world_size={}, args.rank={}".format(args.world_size, args.rank))
    total_batch_size = global_batch_size(args)
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if args.rank == 0 or args.rank == -1:
        print("parsed args:")
        print(args)
    # Prepare optimizer
    (
        model,
        optimizer,
        lr_scheduler,
        checkpoint,
        global_step,
    ) = prepare_model_and_optimizer(args, device)
    worker_seeds, shuffling_seeds = utils.setup_seeds(
        args.seed, args.num_epochs_to_generate_seeds_for, device
    )
    worker_seed = worker_seeds[args.rank]
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    worker_init = WorkerInitObj(worker_seed)
    print("[info] Setting seed: ", args.seed, ". worker seed: ", worker_seed)

    samples_trained = (
        global_step
        * args.train_batch_size
        * args.gradient_accumulation_steps
        * args.world_size
    )
    final_loss = float("inf")
    train_time_raw = float("inf")
    raw_train_start = time.time()

    if args.do_train:
        model.train()
        most_recent_ckpts_paths = []
        average_loss = 0.0  # averaged loss every args.log_freq steps
        epoch = 1
        training_steps = 0
        end_training, converged = False, False
        samples_trained_prev_for_eval = 0
        samples_trained_prev_for_save = 0

        # pre-compute eval boundaries
        samples_trained_per_step = (
            args.train_batch_size * args.gradient_accumulation_steps * args.world_size
        )
        start, stop, step = (
            args.eval_iter_start_samples,
            args.max_samples_termination,
            args.eval_iter_samples,
        )
        eval_steps = [
            math.ceil(i / samples_trained_per_step)
            for i in np.arange(start, stop, step)
        ]
        eval_count = 0
        next_eval_step = eval_steps[eval_count]

        if args.converge:
            print("max samples termination: ", args.max_samples_termination)
            print("samples trained per step = ", samples_trained_per_step)
            print(
                "all eval steps, start: ",
                eval_steps[0],
                ", end: ",
                eval_steps[len(eval_steps) - 1],
                ", eval times: ",
                len(eval_steps),
            )

        if args.target_mlm_accuracy:
            if args.train_mlm_accuracy_window_size > 0:
                accuracy_scores = []
                avg_mlm_accuracy = torch.Tensor([0])

        first_epoch = True

        # resume ckpt
        if found_resume_checkpoint(args) and checkpoint is not None:
            print("[info] construct file from checkpoint")
            global_step = checkpoint["global_step"]
            print("[info] loadind global step... ", global_step)
            f_start_id = checkpoint["files"][0]
            print("[info] loadind samples file start id... ", f_start_id)
            files = checkpoint["files"][1:]
            num_files = len(files)
            print("[info] loadind samples num files... ", num_files)
            samples_trained = checkpoint["samples_trained"]
            print("[info] loadind samples trained... ", samples_trained)
            samples_trained_prev_for_eval = checkpoint["samples_trained_prev_for_eval"]
            print(
                "[info] loadind samples trained prev for eval... ",
                samples_trained_prev_for_eval,
            )
            samples_trained_prev_for_save = checkpoint["samples_trained_prev_for_save"]
            print(
                "[info] loadind samples trained prev for save... ",
                samples_trained_prev_for_save,
            )
            epoch = checkpoint["epoch"]
            print("[info] loadind epoch... ", epoch)
            eval_count = checkpoint["eval_count"]
            print("[info] loadind eval_count... ", eval_count)
            eval_steps = checkpoint["eval_steps"]
            print("[info] loadind eval_steps... ")
            next_eval_step = eval_steps[eval_count]
            print("[info] next eval step... ")
        else:
            print("[info] construct file from initialization")
            print("[info] input dir = ", args.input_dir)
            files = [
                os.path.join(args.input_dir, f)
                for f in os.listdir(args.input_dir)
                if os.path.isfile(os.path.join(args.input_dir, f)) and "part" in f
            ]
            files.sort()
            num_files = len(files)
            print("[info] num files = ", num_files)
            random.Random(shuffling_seeds[epoch % len(shuffling_seeds)]).shuffle(files)
            f_start_id = 0

    sync_clock()

    # outer loop to finish max_steps
    while global_step < args.max_steps and not end_training:
        if args.converge:
            print("global step = ", global_step)
        if args.rank == 0 or args.rank == -1:
            now_time = time.time()
            print("epoch:", epoch)

        # Reshuffle file list on subsequent epochs
        if not first_epoch:
            files = [
                os.path.join(args.input_dir, f)
                for f in os.listdir(args.input_dir)
                if os.path.isfile(os.path.join(args.input_dir, f)) and "part" in f
            ]
            files.sort()
            num_files = len(files)
            random.Random(shuffling_seeds[epoch % len(shuffling_seeds)]).shuffle(files)
            f_start_id = 0
        first_epoch = False

        shared_file_list = {}

        if torch.distributed.is_initialized() and args.world_size > num_files:
            remainder = args.world_size % num_files
            data_file = files[
                (f_start_id * args.world_size + args.rank + remainder * f_start_id)
                % num_files
            ]
            print("rank: ", args.rank, ". data file: ", data_file)
        else:
            data_file = files[(f_start_id * args.world_size + args.rank) % num_files]

        train_data = pretraining_dataset(data_file, args.max_predictions_per_seq)
        if torch.distributed.is_initialized() and not args.no_ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        else:
            train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(
            train_data,
            sampler=train_sampler,
            batch_size=args.train_batch_size,
            num_workers=args.workers,
            pin_memory=True,
        )

        for f_id in range(f_start_id, len(files)):

            # data file for next file loop train
            if args.world_size > num_files:
                data_file = files[
                    (f_id * args.world_size + args.rank + remainder * f_id) % num_files
                ]
            else:
                data_file = files[(f_id * args.world_size + args.rank) % num_files]

            # initial
            gloss = 0.0
            gloss_list = []
            acc_list = []

            # config profiler
            import contextlib

            def profiler_setup(profiling=False, *args, **kwargs):
                if profiling:
                    return torch.profiler.profile(*args, **kwargs)
                else:
                    return contextlib.nullcontext()

            activities = [torch.profiler.ProfilerActivity.CPU]
            if args.device == "xpu":
                activities.append(torch.profiler.ProfilerActivity.XPU)
            elif args.device == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            # fixed wait iters = 1, active iters = 1, warmup_iters = 3, at_least 5 iters
            # more iters will be skipped, and repeat will be fixed to 1
            num_iters = (
                int(args.num_iterations)
                if args.num_iterations
                else len(train_dataloader)
            )
            skip_iters = max(num_iters - 5, 0)
            schedule = torch.profiler.schedule(
                skip_first=skip_iters, wait=1, warmup=3, active=1
            )

            def trace_handle(prof):
                if args.device == "xpu":
                    print(str(prof.key_averages().table(sort_by="self_xpu_time_total")))
                else:
                    print(
                        str(prof.key_averages().table(sort_by="self_cuda_time_total"))
                    )
                if args.export_chrome_trace:
                    print("export_chrome_trace", flush=True)
                    prof.export_chrome_trace("./profile_trace.json")

            # start profiler, or none while profiling is false
            with profiler_setup(
                args.profile,
                activities=activities,
                schedule=schedule,
                on_trace_ready=trace_handle,
                record_shapes=True,
            ) as prof:
                # this file loop train: f_id
                for step, batch in enumerate(train_dataloader):
                    if not args.converge:
                        if training_steps >= 10 + int(args.num_iterations):
                            latency_list = latency_list[10:]
                            avg = sum(latency_list) / len(latency_list)
                            print("bert_train latency:  " + str(avg) + "  s")
                            print(
                                "bert_train throughput:  "
                                + str(total_batch_size / args.world_size / avg)
                                + "  sentences/s"
                            )
                            print("perplexity = " + str(gloss))
                            return
                    training_steps += 1

                    # H2D
                    start_h2d = get_time_sync(args.device)
                    (
                        input_ids,
                        segment_ids,
                        input_mask,
                        masked_lm_labels,
                        next_sentence_labels,
                    ) = batch
                    input_ids = input_ids.to(device)
                    segment_ids = segment_ids.to(device)
                    input_mask = input_mask.to(device)
                    masked_lm_labels = masked_lm_labels.to(device)
                    next_sentence_labels = next_sentence_labels.to(device)
                    end_h2d = get_time_sync(args.device)

                    info = {
                        "input_ids": [input_ids.shape, input_ids.dtype],
                        "segment_ids": [segment_ids.shape, segment_ids.dtype],
                        "input_mask": [input_mask.shape, input_mask.dtype],
                        "masked_lm_labels": [
                            masked_lm_labels.shape,
                            masked_lm_labels.dtype,
                        ],
                        "next_sentence_labels": [
                            next_sentence_labels.shape,
                            next_sentence_labels.dtype,
                        ],
                    }
                    if args.info:
                        print("datainfo", info)

                    def step_training():
                        outputs = None
                        if args.amp:
                            if args.device == "cpu":
                                with torch.autocast(
                                    "cpu",
                                ):
                                    outputs = model(
                                        input_ids=input_ids,
                                        token_type_ids=segment_ids,
                                        attention_mask=input_mask,
                                        labels=masked_lm_labels,
                                        next_sentence_label=next_sentence_labels,
                                    )
                            elif args.device == "xpu":
                                with torch.autocast(
                                    "xpu", enabled=True, dtype=optimize_dtype
                                ):
                                    outputs = model(
                                        input_ids=input_ids,
                                        token_type_ids=segment_ids,
                                        attention_mask=input_mask,
                                        labels=masked_lm_labels,
                                        next_sentence_label=next_sentence_labels,
                                    )
                            elif args.device == "cuda":
                                with torch.autocast(
                                    "cuda",
                                ):
                                    outputs = model(
                                        input_ids=input_ids,
                                        token_type_ids=segment_ids,
                                        attention_mask=input_mask,
                                        labels=masked_lm_labels,
                                        next_sentence_label=next_sentence_labels,
                                    )
                        elif args.fp8:
                            # most used path for current xpu OOB solution
                            fp8_model = prepare_fp8(model)
                            with fp8_autocast(
                                enabled=True, fp8_recipe=DelayedScaling()
                            ):
                                outputs = fp8_model(
                                    input_ids=input_ids,
                                    token_type_ids=segment_ids,
                                    attention_mask=input_mask,
                                    labels=masked_lm_labels,
                                    next_sentence_label=next_sentence_labels,
                                )
                        else:
                            # most used path for current xpu OOB solution
                            with torch.autocast(
                                "xpu", enabled=True, dtype=optimize_dtype
                            ):
                                outputs = model(
                                    input_ids=input_ids,
                                    token_type_ids=segment_ids,
                                    attention_mask=input_mask,
                                    labels=masked_lm_labels,
                                    next_sentence_label=next_sentence_labels,
                                )
                        loss = outputs.loss
                        loss = loss / args.gradient_accumulation_steps
                        loss.backward()

                        # clip grad norm
                        if hasattr(optimizer, "clip_grad_norm_"):
                            ggnorm = optimizer.clip_grad_norm_(1.0)
                        else:
                            ggnorm = torch.nn.utils.clip_grad_norm_(
                                model.parameters(), 1.0
                            )

                        if (
                            step + 1
                        ) % args.gradient_accumulation_steps == 0 or step == len(
                            train_dataloader
                        ) - 1:
                            start_opt = get_time_sync(args.device)
                            optimizer.step()
                            lr_scheduler.step()
                            optimizer.zero_grad()
                            end_opt = get_time_sync(args.device)

                        return outputs, end_opt - start_opt

                    # train and profile
                    if args.device == "xpu":
                        try:
                            import memory_check

                            memory_check.display_mem("xpu:0")
                        except:
                            pass
                        start_training_time = get_time_sync(args.device)
                        outputs, opt_time = step_training()
                    elif args.device == "cuda":
                        start_training_time = get_time_sync(args.device)
                        outputs, opt_time = step_training()
                    end_training_time = get_time_sync(args.device)

                    if args.profile:
                        prof.step()

                    timeinfo = {
                        "h2d": end_h2d - start_h2d,
                        "training": end_training_time - start_training_time,
                        "opt": opt_time,  # training include opt_time
                    }

                    timeinfo["total"] = timeinfo["h2d"] + timeinfo["training"]
                    timeinfo["total"] = timeinfo["training"]
                    latency_list.append(timeinfo["total"] / 1000.0)
                    if args.info:
                        print("timeinfo", timeinfo)

                    # no need to calculate loss and acc of each iter
                    if args.converge and global_step % PRINT_ITER_FACTOR == 0:
                        gloss, lm_acc, num_masked, seq_acc, seq_tot = calc_accuracy(
                            outputs, masked_lm_labels, next_sentence_labels, args
                        )
                        gloss_list.append(gloss)
                        acc_list.append(lm_acc.item())
                        avg_gloss = round(sum(gloss_list) / len(gloss_list), 3)
                        avg_acc = round(sum(acc_list) / len(acc_list), 3)
                        print(
                            "Global Step [",
                            global_step,
                            "/",
                            args.max_steps,
                            "] perf:",
                            str(round(timeinfo["total"] / 1000.0, 4)),
                            " s/it",
                            ", gloss:",
                            round(gloss, 4),
                            "(",
                            avg_gloss,
                            "), lm acc:",
                            round(lm_acc.item(), 4),
                            "(",
                            avg_acc,
                            ")",
                        )
                    elif args.converge:
                        print("Global Step [", global_step, "/", args.max_steps, "]")
                    else:
                        gloss, lm_acc, num_masked, seq_acc, seq_tot = calc_accuracy(
                            outputs, masked_lm_labels, next_sentence_labels, args
                        )

                    if not args.converge:
                        if (
                            args.benchmark_steps > 0
                            and global_step + 1 >= args.benchmark_steps
                        ):
                            synchronize(args.device)
                            if args.rank == 0:
                                print(
                                    f"Done Benchmarking {args.benchmark_steps} steps."
                                )
                            sys.exit(0)

                    if args.info:
                        info = {
                            "gloss": gloss,
                            "lm_acc": lm_acc,
                            "num_masked": num_masked,
                            "seq_acc": seq_acc,
                            "seq_tot": seq_tot,
                        }
                        print("outinfo", info)

                    update_step = training_steps % args.gradient_accumulation_steps == 0
                    if update_step:
                        now_lr = optimizer.param_groups[0]["lr"]
                        global_step += 1
                        if args.converge and global_step % PRINT_ITER_FACTOR == 0:
                            print(
                                "global step = ",
                                global_step,
                                ". next eval step = ",
                                next_eval_step,
                                ". now lr = ",
                                now_lr,
                            )
                        # global_step == next_eval_step, means this step needs do evaluaion
                        if (
                            args.eval_dir
                            and args.eval_iter_samples > 0
                            and global_step == next_eval_step
                        ):
                            # first do validation or load ckpt
                            if eval_count == 0 or args.resume_from_checkpoint:
                                eval_dataloader = create_eval_dataset(
                                    args, worker_init_fn=worker_init
                                )
                            samples_trained = (
                                global_step
                                * args.train_batch_size
                                * args.gradient_accumulation_steps
                                * args.world_size
                            )
                            print(
                                "\n\nBefore eval, samples trained = ", samples_trained
                            )
                            print(
                                "Before eval, samples trained prev for eval = ",
                                samples_trained_prev_for_eval,
                            )

                            # update samples trained prev in validation
                            samples_trained_prev_for_eval = samples_trained

                            print(
                                "This step [", global_step, "] begin to do validation"
                            )
                            eval_avg_loss, eval_avg_mlm_accuracy = run_eval(
                                model,
                                eval_dataloader,
                                device,
                                args.num_eval_examples,
                                args,
                                first_eval=(eval_count == 0),
                            )
                            if args.converge:
                                print(
                                    "Finish validation, this step [",
                                    global_step,
                                    "] loss: ",
                                    round(eval_avg_loss.item(), 4),
                                    " final acc: ",
                                    round(eval_avg_mlm_accuracy.item(), 4),
                                    "\n\n",
                                )

                            if not args.converge and (
                                args.rank == 0 or args.rank == -1
                            ):
                                print(
                                    {
                                        "global_steps": global_step,
                                        "eval_loss": eval_avg_loss,
                                        "eval_mlm_accuracy": eval_avg_mlm_accuracy,
                                    }
                                )

                            if args.converge:
                                # judge convergence or not
                                if args.target_mlm_accuracy:
                                    if (
                                        eval_avg_mlm_accuracy
                                        >= args.target_mlm_accuracy
                                    ):
                                        end_training, converged = True, True
                                        if utils.is_main_process():
                                            print(
                                                "%f > %f, [Successfully] Target MLM Accuracy reached at %d"
                                                % (
                                                    eval_avg_mlm_accuracy,
                                                    args.target_mlm_accuracy,
                                                    global_step,
                                                )
                                            )
                            eval_count += 1
                            next_eval_step = eval_steps[eval_count]

                    # update samples trained each step
                    samples_trained = (
                        global_step
                        * args.train_batch_size
                        * args.gradient_accumulation_steps
                        * args.world_size
                    )

                    # for save/load ckpt
                    if args.converge:
                        if args.converge and global_step % PRINT_ITER_FACTOR == 0:
                            print(
                                "samples trained: ",
                                samples_trained,
                                ". samples trained prev for save: ",
                                samples_trained_prev_for_save,
                                ". this round trained: ",
                                (samples_trained - samples_trained_prev_for_save),
                            )
                        if end_training or (
                            samples_trained - samples_trained_prev_for_save
                            >= args.num_samples_per_checkpoint
                            and samples_trained >= args.min_samples_to_start_checkpoints
                        ):

                            # update samples trained prev in save ckpt
                            samples_trained_prev_for_save = samples_trained

                            # main process save ckpt to save disk space
                            if utils.is_main_process() and not args.skip_checkpoint:
                                print("Prepare to save model")
                                # Save a trained model
                                model_to_save = (
                                    model.module if hasattr(model, "module") else model
                                )  # Only save the model it-self

                                device_info_for_ckpt = "samples_"
                                if torch.distributed.is_initialized():
                                    device_info_for_ckpt = (
                                        "rank_" + str(args.rank) + "_samples_"
                                    )
                                if end_training and converged:
                                    if args.phase2:
                                        output_save_file = os.path.join(
                                            args.output_dir,
                                            "phase2_ckpt_"
                                            + device_info_for_ckpt
                                            + str(samples_trained)
                                            + ".converged.pt",
                                        )
                                    else:
                                        output_save_file = os.path.join(
                                            args.output_dir,
                                            "phase1_ckpt_"
                                            + device_info_for_ckpt
                                            + str(samples_trained)
                                            + ".converged.pt",
                                        )
                                else:
                                    if args.phase2:
                                        output_save_file = os.path.join(
                                            args.output_dir,
                                            "phase2_ckpt_"
                                            + device_info_for_ckpt
                                            + str(samples_trained)
                                            + ".pt",
                                        )
                                    else:
                                        output_save_file = os.path.join(
                                            args.output_dir,
                                            "phase1_ckpt_"
                                            + device_info_for_ckpt
                                            + str(samples_trained)
                                            + ".pt",
                                        )
                                if args.do_train:
                                    print(
                                        "\n\n[info] save checkpoint: ",
                                        output_save_file,
                                        "\n",
                                    )
                                    torch.save(
                                        {
                                            "model": model_to_save.state_dict(),
                                            "optimizer": optimizer.state_dict(),
                                            "lr_scheduler": lr_scheduler.state_dict(),
                                            "files": [f_id] + files,
                                            "epoch": epoch,
                                            "global_step": global_step,
                                            "samples_trained": samples_trained,
                                            "samples_trained_prev_for_eval": samples_trained_prev_for_eval,
                                            "samples_trained_prev_for_save": samples_trained_prev_for_save,
                                            "eval_count": eval_count,
                                            "eval_steps": eval_steps,
                                        },
                                        output_save_file,
                                    )

                                    most_recent_ckpts_paths.append(output_save_file)
                                    if (
                                        len(most_recent_ckpts_paths)
                                        > args.keep_n_most_recent_checkpoints
                                    ):
                                        print(
                                            "\n\nkeep ckpt number:",
                                            args.keep_n_most_recent_checkpoints,
                                        )
                                        ckpt_to_be_removed = (
                                            most_recent_ckpts_paths.pop(0)
                                        )
                                        print(
                                            "remove ckpt:", ckpt_to_be_removed, "\n\n"
                                        )
                                        os.remove(ckpt_to_be_removed)

                            if (
                                samples_trained >= args.max_samples_termination
                                or end_training
                            ):
                                status = "success" if converged else "aborted"
                                end_training = True
                                break

            del train_dataloader
            if args.converge and utils.is_main_process():
                print("\nfinish one file, global step = ", global_step)
                print("finish one file, samples trained = ", samples_trained)
                print(
                    "finish one file, max samples termination = ",
                    args.max_samples_termination,
                    "\n",
                )
            if samples_trained >= args.max_samples_termination or end_training:
                status = "success" if converged else "aborted"
                if utils.is_main_process():
                    print("eventual status = ", status)
                end_training = True
                break
            train_dataloader, data_file = create_pretraining_dataset(
                data_file,
                args.max_predictions_per_seq,
                shared_file_list,
                args,
                worker_init_fn=worker_init,
            )

        epoch += 1

    global_end_time = time.time()
    if utils.is_main_process():
        print(
            "[info] total consume time: ",
            ((global_end_time - global_start_time) / 60.0),
            " min.",
        )
    return args, final_loss, train_time_raw


if __name__ == "__main__":
    main()
