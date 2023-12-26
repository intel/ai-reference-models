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
import modeling_bert_patched

# from accelerate import Accelerator, DistributedType
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForPreTraining,
    SchedulerType,
    get_scheduler,
    set_seed,
)

from pcl_pytorch_extension import bert as pcl_bert
from schedulers import LinearWarmUpScheduler, LinearWarmupPolyDecayScheduler
import mlperf_logger
from lamb import Lamb

from pcl_pytorch_extension.optim import DistLamb

ref_time = 0


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
        rank = args.local_rank
        remainder = args.num_eval_examples % args.world_size
        if rank < remainder:
            return chunk_size + 1
        else:
            return chunk_size


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
        rank = args.local_rank
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
        eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0
    )
    return eval_dataloader


class pretraining_dataset(Dataset):
    def __init__(self, input_file, max_pred_length):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        f = h5py.File(input_file, "r")
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
            torch.from_numpy(input[index].astype(np.int64))
            if indice < 5
            else torch.from_numpy(np.asarray(input[index].astype(np.int64)))
            for indice, input in enumerate(self.inputs)
        ]
        masked_lm_labels = torch.zeros(input_ids.shape, dtype=torch.long) - 100
        index = self.max_pred_length
        masked_token_count = torch.count_nonzero(masked_lm_positions)
        if masked_token_count != 0:
            index = masked_token_count
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]

        return [
            input_ids,
            segment_ids,
            input_mask,
            masked_lm_labels,
            next_sentence_labels,
        ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a Masked Language Modeling task"
    )

    ## Required parameters
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
        default=-1,
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
        "--init_checkpoint",
        default=None,
        type=str,
        help="The initial checkpoint to start training from.",
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
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
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
        "--max_predictions_per_seq",
        default=76,
        type=int,
        help="The maximum total of masked tokens in input sequence",
    )
    parser.add_argument(
        "--train_batch_size",
        default=18,
        type=int,
        help="Total batch size for training.",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=128,
        type=int,
        help="Total batch size for training.",
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
        default=1536,
        type=float,
        help="Total number of training steps to perform.",
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
        help="Whether to resume training from checkpoint. If set, precedes init_checkpoint/init_tf_checkpoint",
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
        "--skip_checkpoint",
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
        "--dist_lamb",
        default=False,
        action="store_true",
        help="Whether to use DistLamb from pcl_bert",
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
        "--seed", type=int, default=42, help="A seed for reproducible training."
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
        "--local_rank", default=0, type=int, help="Total batch size for training."
    )
    parser.add_argument(
        "--world_size", default=1, type=int, help="Total batch size for training."
    )

    parser.add_argument(
        "--use_pcl",
        action="store_true",
        help="Whether to use PCL Fused impl when available",
    )
    parser.add_argument(
        "--pcl_bf16", action="store_true", help="Whether to use PCL BF16 impl"
    )
    parser.add_argument(
        "--profile", action="store_true", help="Whether to enable profiling"
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
        "--multi_instance",
        action="store_true",
        help="Skip torch.distributed initialization to Run multiple instance independently",
    )
    parser.add_argument(
        "--dist_profile",
        action="store_true",
        help="Whether to enable distributed timing profile",
    )

    args = parser.parse_args()

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    # assert args.init_checkpoint is not None or args.init_tf_checkpoint is not None or found_resume_checkpoint(args), \
    #    "Must specify --init_checkpoint, --init_tf_checkpoint or have ckpt to resume from in --output_dir of the form *.pt"

    # assert not (args.init_checkpoint is not None and args.init_tf_checkpoint is not None), \
    #        "Can only specify one of --init_checkpoint and --init_tf_checkpoint"
    return args


def found_resume_checkpoint(args):
    if args.phase2:
        checkpoint_str = "phase2_ckpt*.pt"
    else:
        checkpoint_str = "phase1_ckpt*.pt"
    return (
        args.resume_from_checkpoint
        and len(glob.glob(os.path.join(args.output_dir, checkpoint_str))) > 0
    )


def setup_training(args):
    device = torch.device("cpu")
    if int(os.environ.get("PMI_SIZE", "0")) > 1 and not args.multi_instance:
        if args.dist_backend == "ccl":
            try:
                import torch_ccl
            except:
                print("CCL backend requested but import torch_ccl failed")
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

        os.environ["RANK"] = os.environ.get("PMI_RANK", "0")
        os.environ["WORLD_SIZE"] = os.environ.get("PMI_SIZE", "1")
        torch.distributed.init_process_group(backend=args.dist_backend)
        device = torch.device("cpu")
        args.local_rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()
        if args.local_rank == 0:
            print(
                f"##################Using {args.dist_backend.upper()} dist run with {args.world_size} ranks",
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


def prepare_model_and_optimizer(args, device):
    global_step = 0
    args.resume_step = 0
    checkpoint = None
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    config.dense_seq_output = args.dense_seq_output
    with pcl_bert.pcl_impl(args.use_pcl, args.pcl_bf16, args.unpad):
        if args.model_name_or_path:
            model = AutoModelForPreTraining.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
            )
        else:
            logger.info("Training new model from scratch")
            model = AutoModelForPreTraining.from_config(config)
    if args.use_pcl:
        pcl_bert.block(model)

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
    if args.dist_lamb:
        optimizer = DistLamb(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            betas=(args.opt_lamb_beta_1, args.opt_lamb_beta_2),
        )
    else:
        optimizer = Lamb(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            betas=(args.opt_lamb_beta_1, args.opt_lamb_beta_2),
        )

    mlperf_logger.log_event(
        key=mlperf_logger.constants.OPT_BASE_LR, value=args.learning_rate, sync=False
    )
    mlperf_logger.log_event(
        key="opt_epsilon", value=optimizer.defaults["eps"], sync=False
    )
    b1, b2 = optimizer.defaults["betas"]
    mlperf_logger.log_event(key="opt_lamb_beta_1", value=b1, sync=False)
    mlperf_logger.log_event(key="opt_lamb_beta_2", value=b2, sync=False)
    mlperf_logger.log_event(
        key="opt_lamb_weight_decay_rate",
        value=optimizer.defaults["weight_decay"],
        sync=False,
    )

    if args.warmup_steps == 0:
        warmup_steps = int(args.max_steps * args.warmup_proportion)
        warmup_start = 0
    else:
        warmup_steps = args.warmup_steps
        warmup_start = args.start_warmup_step

    lr_scheduler = LinearWarmupPolyDecayScheduler(
        optimizer,
        start_warmup_steps=warmup_start,
        warmup_steps=warmup_steps,
        total_steps=args.max_steps,
        end_learning_rate=0.0,
        degree=1.0,
    )
    return model, optimizer, lr_scheduler, checkpoint, global_step


def take_optimizer_step(args, optimizer, model, overflow_buf, global_step):
    global skipped_steps
    optimizer.step()
    global_step += 1
    return global_step


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
        for batch in eval_dataloader:
            (
                input_ids,
                segment_ids,
                input_mask,
                masked_lm_labels,
                next_sentence_labels,
            ) = batch
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


def exchange_padding_fast(
    input_ids,
    segment_ids,
    input_mask,
    masked_lm_labels,
    next_sentence_labels,
    max_batch_size,
):
    device = "cpu"
    comm_dtype = torch.int32
    pad_size = max_batch_size - input_ids.shape[0]
    if pad_size > 0:
        input_ids = F.pad(input_ids, (0, 0, 0, pad_size))
        segment_ids = F.pad(segment_ids, (0, 0, 0, pad_size))
        input_mask = F.pad(input_mask, (0, 0, 0, pad_size))
        masked_lm_labels = F.pad(masked_lm_labels, (0, 0, 0, pad_size))
        next_sentence_labels = F.pad(next_sentence_labels, (0, pad_size))
    nranks = torch.distributed.get_world_size()
    nseqs = input_mask.shape[0]
    ntokensperseq = input_mask.shape[1]
    my_rank = torch.distributed.get_rank()

    flattened_length_seq = nseqs * ntokensperseq
    flattened_length_nsp = nseqs

    def get_local_packet_size():
        return 4 * flattened_length_seq + flattened_length_nsp

    # Storing tensors in same order as arguments
    def encode_packet(
        input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels
    ):

        packet = torch.zeros([get_local_packet_size()], device=device, dtype=comm_dtype)

        curr_pos = 0

        packet[curr_pos : curr_pos + flattened_length_seq] = input_ids.view(-1)[:]
        curr_pos += flattened_length_seq

        packet[curr_pos : curr_pos + flattened_length_seq] = segment_ids.view(-1)[:]
        curr_pos += flattened_length_seq

        packet[curr_pos : curr_pos + flattened_length_seq] = input_mask.view(-1)[:]
        curr_pos += flattened_length_seq

        packet[curr_pos : curr_pos + flattened_length_seq] = masked_lm_labels.view(-1)[
            :
        ]
        curr_pos += flattened_length_seq

        packet[curr_pos : curr_pos + flattened_length_nsp] = next_sentence_labels.view(
            -1
        )[:]

        return packet

    def decode_packet(flat_packet):
        packet = flat_packet.view(nranks, get_local_packet_size())

        curr_pos = 0

        input_ids_ = (
            packet[:, curr_pos : curr_pos + flattened_length_seq]
            .contiguous()
            .view(nranks, nseqs, ntokensperseq)
        )
        curr_pos += flattened_length_seq

        segment_ids_ = (
            packet[:, curr_pos : curr_pos + flattened_length_seq]
            .contiguous()
            .view(nranks, nseqs, ntokensperseq)
        )
        curr_pos += flattened_length_seq

        input_mask_ = (
            packet[:, curr_pos : curr_pos + flattened_length_seq]
            .contiguous()
            .view(nranks, nseqs, ntokensperseq)
        )
        curr_pos += flattened_length_seq

        masked_lm_labels_ = (
            packet[:, curr_pos : curr_pos + flattened_length_seq]
            .contiguous()
            .view(nranks, nseqs, ntokensperseq)
        )
        curr_pos += flattened_length_seq

        next_sentence_labels_ = (
            packet[:, curr_pos : curr_pos + flattened_length_nsp]
            .contiguous()
            .view(nranks, nseqs)
        )

        return (
            input_ids_,
            segment_ids_,
            input_mask_,
            masked_lm_labels_,
            next_sentence_labels_,
        )

    tensors = encode_packet(
        input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels
    )

    tensors_ = torch.zeros(
        [nranks, get_local_packet_size()], device=device, dtype=comm_dtype
    )
    tensors_ = list(torch.split(tensors_, 1))

    # torch.distributed.all_gather(tensors_, tensors.view(torch.float16))
    torch.distributed.all_gather(tensors_, tensors)

    tensors_ = torch.stack(tensors_).view(comm_dtype).long()
    (
        input_ids_,
        segment_ids_,
        input_mask_,
        masked_lm_labels_,
        next_sentence_labels_,
    ) = decode_packet(tensors_)

    seqlens_, indices = torch.sort(input_mask_.sum(dim=2).view(-1), descending=True)

    if pad_size > 0:
        input_ids_sorted = input_ids_.view(nranks * nseqs, ntokensperseq)[indices[:], :]
        segment_ids_sorted = segment_ids_.view(nranks * nseqs, ntokensperseq)[
            indices[:], :
        ]
        input_mask_sorted = input_mask_.view(nranks * nseqs, ntokensperseq)[
            indices[:], :
        ]
        masked_lm_labels_sorted = masked_lm_labels_.view(nranks * nseqs, ntokensperseq)[
            indices[:], :
        ]
        next_sentence_labels_sorted = next_sentence_labels_.view(nranks * nseqs)[
            indices[:]
        ]
        # we need to remove the empty sequences we added to the batch
        valid_idx = seqlens_.view(nseqs, nranks)[:, my_rank] > 0
        input_ids_sorted = input_ids_sorted.view(nseqs, nranks, ntokensperseq)[
            valid_idx, my_rank, :
        ].contiguous()
        segment_ids_sorted = segment_ids_sorted.view(nseqs, nranks, ntokensperseq)[
            valid_idx, my_rank, :
        ].contiguous()
        input_mask_sorted = input_mask_sorted.view(nseqs, nranks, ntokensperseq)[
            valid_idx, my_rank, :
        ].contiguous()
        masked_lm_labels_sorted = masked_lm_labels_sorted.view(
            nseqs, nranks, ntokensperseq
        )[valid_idx, my_rank, :].contiguous()
        next_sentence_labels_sorted = next_sentence_labels_sorted.view(nseqs, nranks)[
            valid_idx, my_rank
        ].contiguous()
    else:
        indices_ = indices.view(nseqs, nranks)[:, my_rank]
        input_ids_sorted = input_ids_.view(nseqs * nranks, ntokensperseq)[
            indices_, :
        ].contiguous()
        segment_ids_sorted = segment_ids_.view(nseqs * nranks, ntokensperseq)[
            indices_, :
        ].contiguous()
        input_mask_sorted = input_mask_.view(nseqs * nranks, ntokensperseq)[
            indices_, :
        ].contiguous()
        masked_lm_labels_sorted = masked_lm_labels_.view(nseqs * nranks, ntokensperseq)[
            indices_, :
        ].contiguous()
        next_sentence_labels_sorted = next_sentence_labels_.view(nseqs * nranks)[
            indices_
        ].contiguous()

    return (
        input_ids_sorted,
        segment_ids_sorted,
        input_mask_sorted,
        masked_lm_labels_sorted,
        next_sentence_labels_sorted,
    )


def main():
    args = parse_args()
    status = "aborted"  # later set to 'success' if termination criteria met
    mlperf_logger.log_start(
        key=mlperf_logger.constants.INIT_START, log_all_ranks=True, sync=False
    )
    device, args = setup_training(args)
    print("####################args.world_size={}".format(args.world_size))
    total_batch_size = global_batch_size(args)
    mlperf_logger.mlperf_submission_log("bert")

    mlperf_logger.log_event(
        key=mlperf_logger.constants.SEED, value=args.seed, sync=False
    )
    mlperf_logger.log_event(
        key=mlperf_logger.constants.GLOBAL_BATCH_SIZE,
        value=global_batch_size(args),
        sync=False,
    )
    mlperf_logger.log_event(key="d_batch_size", value=args.train_batch_size, sync=False)
    mlperf_logger.log_event(
        key=mlperf_logger.constants.GRADIENT_ACCUMULATION_STEPS,
        value=args.gradient_accumulation_steps,
        sync=False,
    )
    mlperf_logger.log_event(
        key="max_predictions_per_seq", value=args.max_predictions_per_seq, sync=False
    )
    mlperf_logger.log_event(
        key="opt_learning_rate_training_steps", value=args.max_steps, sync=False
    )
    mlperf_logger.log_event(
        key="num_warmup_steps",
        value=int(args.warmup_proportion * args.max_steps)
        if args.warmup_steps == 0
        else args.warmup_steps,
        sync=False,
    )
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if args.local_rank == 0 or args.local_rank == -1:
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
    worker_seed = worker_seeds[args.local_rank]

    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    worker_init = WorkerInitObj(worker_seed)
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
        samples_trained_prev = 0

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
        # pool = ProcessPoolExecutor(1)

        if args.target_mlm_accuracy:
            if args.train_mlm_accuracy_window_size > 0:
                accuracy_scores = []
                avg_mlm_accuracy = torch.Tensor([0])

        first_epoch = True
        if found_resume_checkpoint(args):
            f_start_id = checkpoint["files"][0]
            files = checkpoint["files"][1:]
            num_files = len(files)
        else:
            files = [
                os.path.join(args.input_dir, f)
                for f in os.listdir(args.input_dir)
                if os.path.isfile(os.path.join(args.input_dir, f)) and "part" in f
            ]
            files.sort()
            num_files = len(files)
            random.Random(shuffling_seeds[epoch]).shuffle(files)
            f_start_id = 0
    global skipped_steps
    if torch.distributed.is_initialized() and not args.no_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            # bucket_cap_mb=8192,
            gradient_as_bucket_view=args.use_gradient_as_bucket_view,
        )

    mlperf_logger.log_end(key=mlperf_logger.constants.INIT_STOP, sync=False)
    mlperf_logger.log_start(key=mlperf_logger.constants.RUN_START, sync=True)
    mlperf_logger.barrier()

    now_step, now_skipped, skip_interval = 0, 0, 0
    # Start prefetching eval dataset
    # if args.eval_dir:
    #    eval_dataset_future = pool.submit(create_eval_dataset, args, worker_init_fn=worker_init)
    # comparing to number of samples in a shard. There are ~38k samples in 4096-way shard, comparing to 10k to be safe
    need_next_training_shard = (
        args.train_batch_size * args.gradient_accumulation_steps * args.max_steps
        > 10000
    )

    sync_clock()

    while global_step < args.max_steps and not end_training:
        mlperf_logger.log_start(
            key=mlperf_logger.constants.EPOCH_START,
            metadata={"epoch_num": epoch},
            sync=False,
        )
        mlperf_logger.log_start(
            key=mlperf_logger.constants.BLOCK_START,
            metadata={"first_epoch_num": epoch, "epoch_count": 1},
            sync=False,
        )
        if args.local_rank == 0 or args.local_rank == -1:
            now_time = time.time()
            print("epoch:", epoch)

        thread = None

        # Reshuffle file list on subsequent epochs
        if not first_epoch:
            files = [
                os.path.join(args.input_dir, f)
                for f in os.listdir(args.input_dir)
                if os.path.isfile(os.path.join(args.input_dir, f)) and "part" in f
            ]
            files.sort()
            num_files = len(files)
            random.Random(shuffling_seeds[epoch]).shuffle(files)
            f_start_id = 0

        first_epoch = False

        shared_file_list = {}

        if torch.distributed.is_initialized() and args.world_size > num_files:
            remainder = args.world_size % num_files
            data_file = files[
                (
                    f_start_id * args.world_size
                    + args.local_rank
                    + remainder * f_start_id
                )
                % num_files
            ]
        else:
            data_file = files[
                (f_start_id * args.world_size + args.local_rank) % num_files
            ]

        # data_file = '/work/ddkalamk/bert/dataset/2048_shards_uncompressed/part_453_of_2048.hdf5' #files[0]
        previous_file = data_file

        train_data = pretraining_dataset(data_file, args.max_predictions_per_seq)
        train_sampler = RandomSampler(train_data)
        # train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=args.train_batch_size
        )
        send_lr_in_parallel = False
        lr_cpu = torch.tensor([0.0], dtype=torch.float32, device="cpu")
        for f_id in range(f_start_id + 1, len(files)):
            if args.world_size > num_files:
                data_file = files[
                    (f_id * args.world_size + args.local_rank + remainder * f_id)
                    % num_files
                ]
            else:
                data_file = files[
                    (f_id * args.world_size + args.local_rank) % num_files
                ]

            previous_file = data_file
            if need_next_training_shard:
                # dataset_future = pool.submit(create_pretraining_dataset, data_file, args.max_predictions_per_seq, shared_file_list, args, worker_init_fn=worker_init)
                pass
            t0 = get_time()
            for step, batch in enumerate(train_dataloader):
                training_steps += 1
                if args.profile and args.use_pcl:
                    pcl_bert.reset_debug_timers()
                t1 = get_time()
                (
                    input_ids,
                    segment_ids,
                    input_mask,
                    masked_lm_labels,
                    next_sentence_labels,
                ) = batch
                if (
                    args.world_size > 1
                    and args.unpad
                    and not args.skip_exchange_padding
                ):
                    (
                        input_ids,
                        segment_ids,
                        input_mask,
                        masked_lm_labels,
                        next_sentence_labels,
                    ) = exchange_padding_fast(
                        input_ids,
                        segment_ids,
                        input_mask,
                        masked_lm_labels,
                        next_sentence_labels,
                        args.train_batch_size,
                    )
                # print(f"Input shape: {batch['input_ids'].shape}")
                t2 = get_time()
                outputs = model(
                    input_ids=input_ids,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                    labels=masked_lm_labels,
                    next_sentence_label=next_sentence_labels,
                )
                t3 = get_time()
                loss = outputs.loss
                # loss = loss / args.gradient_accumulation_steps
                if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(
                    train_dataloader
                ) - 1:
                    loss.backward()
                    t4 = get_time()
                    if hasattr(optimizer, "clip_grad_norm_"):
                        ggnorm = optimizer.clip_grad_norm_(1.0)
                    else:
                        ggnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    t40 = get_time()
                    mlperf_logger.barrier()
                    t41 = get_time()
                    if args.no_ddp:
                        optimizer.sync_grads()
                    t42 = get_time()
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    # progress_bar.update(1)
                else:
                    # with model.no_sync():
                    loss.backward()
                    t4 = get_time()
                    if hasattr(optimizer, "clip_grad_norm_"):
                        ggnorm = optimizer.clip_grad_norm_(1.0)
                    else:
                        ggnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    t40 = get_time()
                    mlperf_logger.barrier()
                    t42 = t41 = get_time()
                    optimizer.acc_and_zero_grad()
                # mlperf_logger.barrier()
                t5 = get_time()
                gloss, lm_acc, num_masked, seq_acc, seq_tot = calc_accuracy(
                    outputs, masked_lm_labels, next_sentence_labels, args
                )
                data_time = t1 - t0
                xchg_time = t2 - t1
                fwd_time = t3 - t2
                bwd_time = t4 - t3
                clip_time = t40 - t4
                sync_time = t41 - t40
                allr_time = t42 - t41
                opt_time = t5 - t42
                total_time = t5 - t0
                nnz = input_mask.sum()

                if (
                    args.dist_profile
                    and torch.distributed.is_initialized()
                    and args.world_size > 1
                ):
                    l_time = [
                        nnz,
                        fwd_time,
                        bwd_time,
                        fwd_time + bwd_time,
                        sync_time,
                        allr_time,
                        total_time,
                        t40,
                        t41,
                        t42,
                    ]
                    t_time = torch.tensor(l_time)
                    t_all_time = torch.empty([args.world_size, len(l_time)])
                    l_all_time = list(t_all_time.split(1))
                    torch.distributed.all_gather(l_all_time, t_time)

                if args.local_rank == 0:
                    print(
                        f"Step {training_steps:5d}: loss: {gloss:6.3f} lm_acc: {lm_acc:.3f} seq_acc: {seq_acc:.3f} lbs: {args.train_batch_size} gbs: {total_batch_size} DT: {data_time:.1f} XT: {xchg_time:.1f} FT: {fwd_time:.1f} BT: {bwd_time:.1f} OT: {opt_time:.1f} TT: {total_time:.1f} LR: {optimizer.param_groups[0]['lr']:.3e} GNorm: {ggnorm:.2f} ST: {sync_time:.1f} AR: {allr_time:.1f} GC: {clip_time:.1f}"
                    )
                    if (
                        args.dist_profile
                        and torch.distributed.is_initialized()
                        and args.world_size > 1
                    ):

                        def print_summary(desc, t):
                            if isinstance(t, torch.Tensor):
                                t = t.tolist()
                            tmin = min(t)
                            tmin_i = t.index(tmin)
                            tmax = max(t)
                            tmax_i = t.index(tmax)
                            tavg = sum(t) / len(t)
                            print(
                                f"{desc} min: {tmin:10.1f} ({tmin_i:2d}) max: {tmax:10.1f} ({tmax_i:2d})  diff: {tmax-tmin:10.1f} ({(tmax-tmin)*100.0/tmin:5.1f}) avg: {tavg:10.1f}"
                            )

                        print_summary("NNZ:", t_all_time[:, 0])
                        print_summary("FWD:", t_all_time[:, 1])
                        print_summary("BWD:", t_all_time[:, 2])
                        print_summary("FBW:", t_all_time[:, 3])
                        print_summary("SYN:", t_all_time[:, 4])
                        print_summary("SYS:", t_all_time[:, 7])
                        print_summary("ARS:", t_all_time[:, 8])
                        print_summary("ARE:", t_all_time[:, 9])
                if args.local_rank == 0 and args.profile and args.use_pcl:
                    pcl_bert.print_debug_timers()
                if args.benchmark_steps > 0 and global_step + 1 >= args.benchmark_steps:
                    mlperf_logger.barrier()
                    if args.local_rank == 0:
                        print(f"Done Benchmarking {args.benchmark_steps} steps.")
                    sys.exit(0)

                update_step = training_steps % args.gradient_accumulation_steps == 0
                divisor = args.gradient_accumulation_steps
                if args.log_freq > 0:
                    average_loss += loss.item()
                if update_step:
                    now_lr = optimizer.param_groups[0]["lr"]
                    # optimizer.step()
                    global_step += 1
                    if (
                        args.eval_dir
                        and args.eval_iter_samples > 0
                        and global_step == next_eval_step
                    ):
                        # on first eval, get eval_dataloader
                        if eval_count == 0:
                            eval_dataloader = create_eval_dataset(
                                args, worker_init_fn=worker_init
                            )  # eval_dataset_future.result(timeout=None)
                        samples_trained = (
                            global_step
                            * args.train_batch_size
                            * args.gradient_accumulation_steps
                            * args.world_size
                        )
                        samples_trained_prev = samples_trained
                        eval_avg_loss, eval_avg_mlm_accuracy = run_eval(
                            model,
                            eval_dataloader,
                            device,
                            args.num_eval_examples,
                            args,
                            first_eval=(eval_count == 0),
                        )
                        if args.local_rank == 0 or args.local_rank == -1:
                            mlperf_logger.log_event(
                                key=mlperf_logger.constants.EVAL_ACCURACY,
                                value=eval_avg_mlm_accuracy.numpy(),
                                metadata={"epoch_num": epoch},
                                sync=False,
                            )
                            print(
                                {
                                    "global_steps": global_step,
                                    "eval_loss": eval_avg_loss,
                                    "eval_mlm_accuracy": eval_avg_mlm_accuracy,
                                }
                            )

                        if args.target_mlm_accuracy:
                            if eval_avg_mlm_accuracy >= args.target_mlm_accuracy:
                                end_training, converged = True, True
                                if utils.is_main_process():
                                    print(
                                        "%f > %f, Target MLM Accuracy reached at %d"
                                        % (
                                            eval_avg_mlm_accuracy,
                                            args.target_mlm_accuracy,
                                            global_step,
                                        )
                                    )

                        eval_count += 1
                        next_eval_step = eval_steps[eval_count]
                if args.target_mlm_accuracy and args.train_mlm_accuracy_window_size > 0:
                    accuracy_scores.append(mlm_acc)
                    if update_step:
                        accuracy_scores = accuracy_scores[
                            -args.train_mlm_accuracy_window_size
                            * args.gradient_accumulation_steps :
                        ]
                        avg_mlm_accuracy[0] = sum(accuracy_scores) / len(
                            accuracy_scores
                        )
                        torch.distributed.all_reduce(
                            avg_mlm_accuracy, op=torch.distributed.ReduceOp.SUM
                        )
                        avg_mlm_accuracy /= args.world_size

                if (
                    args.log_freq > 0
                    and training_steps
                    % (args.log_freq * args.gradient_accumulation_steps)
                    == 0
                ):
                    samples_trained = (
                        global_step
                        * args.train_batch_size
                        * args.gradient_accumulation_steps
                        * args.world_size
                    )
                    if args.local_rank == 0 or args.local_rank == -1:
                        time_interval = time.time() - now_time
                        step_interval = global_step - now_step
                        now_time = time.time()
                        now_step = global_step
                        training_perf = (
                            args.train_batch_size
                            * args.gradient_accumulation_steps
                            * args.world_size
                            * (step_interval + skip_interval)
                            / time_interval
                        )
                        skip_interval = 0

                        if args.train_mlm_accuracy_window_size > 0:
                            print(
                                {
                                    "training_steps": training_steps,
                                    "average_loss": average_loss
                                    / (args.log_freq * divisor),
                                    "step_loss": loss.item()
                                    * args.gradient_accumulation_steps
                                    / divisor,
                                    "learning_rate": now_lr,
                                    "seq/s": training_perf,
                                    "global_steps": now_step,
                                    "samples_trained": samples_trained,
                                    "skipped_steps": now_skipped,
                                    "timestamp": now_time,
                                    "mlm_accuracy": avg_mlm_accuracy[0].item(),
                                }
                            )
                        else:
                            print(
                                {
                                    "training_steps": training_steps,
                                    "average_loss": average_loss
                                    / (args.log_freq * divisor),
                                    "step_loss": loss.item()
                                    * args.gradient_accumulation_steps
                                    / divisor,
                                    "learning_rate": now_lr,
                                    "seq/s": training_perf,
                                    "global_steps": now_step,
                                    "samples_trained": samples_trained,
                                    "skipped_steps": now_skipped,
                                    "timestamp": now_time,
                                }
                            )

                        # for DLFW CI/CD
                        mlperf_logger.log_event(
                            key="tracked_stats",
                            value={
                                "seq/sec": training_perf,
                                "step_loss": loss.item()
                                * args.gradient_accumulation_steps
                                / divisor,
                                "avg_loss": average_loss / (args.log_freq * divisor),
                                "lr": now_lr,
                            },
                            metadata={"step": (epoch, training_steps)},
                            sync=False,
                        )

                        mlperf_logger.log_event(key="throughput", value=training_perf)

                    average_loss = 0

                if global_step >= args.max_steps or end_training:
                    status = "success" if converged else "aborted"
                    end_training = True
                    train_time_raw = time.time() - raw_train_start
                    average_loss = torch.tensor(average_loss, dtype=torch.float32)
                    if args.log_freq > 0:
                        last_num_steps = (
                            int(training_steps / args.gradient_accumulation_steps)
                            % args.log_freq
                        )
                        last_num_steps = (
                            args.log_freq if last_num_steps == 0 else last_num_steps
                        )
                        average_loss = average_loss / (last_num_steps * divisor)
                    if torch.distributed.is_initialized():
                        average_loss /= args.world_size
                        torch.distributed.all_reduce(average_loss)
                    final_loss = average_loss.item()
                    if utils.is_main_process():
                        if args.train_mlm_accuracy_window_size > 0:
                            print(
                                (
                                    epoch,
                                    training_steps / args.gradient_accumulation_steps,
                                ),
                                {
                                    "final_loss": final_loss,
                                    "final_mlm_accuracy": avg_mlm_accuracy[0].item(),
                                },
                            )
                        else:
                            print(
                                (
                                    epoch,
                                    training_steps / args.gradient_accumulation_steps,
                                ),
                                {"final_loss": final_loss},
                            )

                if end_training or (
                    samples_trained - samples_trained_prev
                    >= args.num_samples_per_checkpoint
                    and samples_trained >= args.min_samples_to_start_checkpoints
                ):
                    samples_trained_prev = samples_trained
                    if utils.is_main_process() and not args.skip_checkpoint:
                        # Save a trained model
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )  # Only save the model it-self
                        if args.phase2:
                            output_save_file = os.path.join(
                                args.output_dir,
                                "phase2_ckpt_{}.pt".format(samples_trained),
                            )
                        else:
                            output_save_file = os.path.join(
                                args.output_dir,
                                "phase1_ckpt_{}.pt".format(samples_trained),
                            )
                        if args.do_train:
                            torch.save(
                                {
                                    "model": model_to_save.state_dict(),
                                    "optimizer": optimizer.state_dict(),
                                    "files": [f_id] + files,
                                },
                                output_save_file,
                            )

                            most_recent_ckpts_paths.append(output_save_file)
                            if (
                                len(most_recent_ckpts_paths)
                                > args.keep_n_most_recent_checkpoints
                            ):
                                ckpt_to_be_removed = most_recent_ckpts_paths.pop(0)
                                os.remove(ckpt_to_be_removed)

                    if samples_trained >= args.max_samples_termination or end_training:
                        status = "success" if converged else "aborted"
                        end_training = True
                        break
                t0 = get_time()

            del train_dataloader

            if samples_trained >= args.max_samples_termination or end_training:
                status = "success" if converged else "aborted"
                end_training = True
                break

            if not need_next_training_shard:
                # dataset_future = pool.submit(create_pretraining_dataset, data_file, args.max_predictions_per_seq, shared_file_list, args, worker_init_fn=worker_init)
                pass
            # train_dataloader, data_file = dataset_future.result(timeout=None)
            train_dataloader, data_file = create_pretraining_dataset(
                data_file,
                args.max_predictions_per_seq,
                shared_file_list,
                args,
                worker_init_fn=worker_init,
            )
        mlperf_logger.log_end(
            key=mlperf_logger.constants.BLOCK_STOP,
            metadata={"first_epoch_num": epoch},
            sync=False,
        )
        mlperf_logger.log_end(
            key=mlperf_logger.constants.EPOCH_STOP,
            metadata={"epoch_num": epoch},
            sync=False,
        )
        epoch += 1

    mlperf_logger.log_event(
        key=mlperf_logger.constants.TRAIN_SAMPLES, value=samples_trained, sync=False
    )
    mlperf_logger.log_event(
        key=mlperf_logger.constants.EVAL_SAMPLES,
        value=args.num_eval_examples,
        sync=False,
    )
    mlperf_logger.log_end(
        key=mlperf_logger.constants.RUN_STOP, metadata={"status": status}, sync=False
    )
    return args, final_loss, train_time_raw


if __name__ == "__main__":
    main()
