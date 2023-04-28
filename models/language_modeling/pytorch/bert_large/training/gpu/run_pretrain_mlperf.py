#!/usr/bin/env python
# coding=utf-8
# Copyright (c) 2022-2023 Intel Corporation
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
except:
    ipex = None

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


def synchronize(device='cpu'):
    if device == 'cuda':
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(torch.cuda.FloatTensor(1))
            torch.cuda.synchronize()
        else:
            torch.cuda.synchronize()
    elif device=='xpu':
        torch.xpu.synchronize()


def get_time_sync(device='cpu'):
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
        rank = args.local_rank
        remainder = args.num_eval_examples % args.world_size
        if rank < remainder:
            return chunk_size + 1
        else:
            return chunk_size


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


def create_pretraining_dataset(input_file, max_pred_length, shared_list, args, worker_init_fn):
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a Masked Language Modeling task"
    )

    ## Required parameters
    parser.add_argument("--input_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain .hdf5 files  for the task.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--eval_dir",
                        default=None,
                        type=str,
                        help="The eval data dir. Should contain .hdf5 files  for the task.")
    parser.add_argument("--eval_iter_start_samples",
                        default=3000000,
                        type=int,
                        help="Sample to begin performing eval.")
    parser.add_argument("--eval_iter_samples",
                        default=16,
                        type=int,
                        help="If set to -1, disable eval, \
                        else evaluate every eval_iter_samples during training")
    parser.add_argument("--num_eval_examples",
                        default=10000,
                        type=int,
                        help="number of eval examples to run eval on")
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        help="The initial checkpoint to start training from.")
    parser.add_argument("--init_tf_checkpoint",
                        default=None,
                        type=str,
                        help="The initial TF checkpoint to start training from.")
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
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
    parser.add_argument("--max_predictions_per_seq",
                        default=76,
                        type=int,
                        help="The maximum total of masked tokens in input sequence")
    parser.add_argument("--train_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--weight_decay_rate",
                        default=0.01,
                        type=float,
                        help="weight decay rate for LAMB.")
    parser.add_argument("--opt_lamb_beta_1",
                        default=0.9,
                        type=float,
                        help="LAMB beta1.")
    parser.add_argument("--opt_lamb_beta_2",
                        default=0.999,
                        type=float,
                        help="LAMB beta2.")
    parser.add_argument("--max_steps",
                        default=1536,
                        type=float,
                        help="Total number of training steps to perform.")
    parser.add_argument("--max_samples_termination",
                        default=14000000,
                        type=float,
                        help="Total number of training samples to run.")
    parser.add_argument("--warmup_proportion",
                        default=0.01,
                        type=float,
                        help="Proportion of optimizer update steps to perform linear learning rate warmup for. "
                             "Typically 1/8th of steps for Phase2")
    parser.add_argument("--warmup_steps",
                        default=0,
                        type=float,
                        help="Number of optimizer update steps to perform linear learning rate warmup for. "
                             "Typically 1/8th of steps for Phase2")
    parser.add_argument("--start_warmup_step",
                        default=0,
                        type=float,
                        help="Starting step for warmup. ")
    parser.add_argument('--log_freq',
                        type=float, 
                        default=10000.0,
                        help='frequency of logging loss. If not positive, no logging is provided for training loss')
    parser.add_argument('--checkpoint_activations',
                        default=False,
                        action='store_true',
                        help="Whether to use gradient checkpointing")
    parser.add_argument("--resume_from_checkpoint",
                        default=False,
                        action='store_true',
                        help="Whether to resume training from checkpoint. If set, precedes init_checkpoint/init_tf_checkpoint")
    parser.add_argument('--keep_n_most_recent_checkpoints',
                        type=int,
                        default=20,
                        help="Number of checkpoints to keep (rolling basis).")
    parser.add_argument('--num_samples_per_checkpoint',
                        type=int,
                        default=500000,
                        help="Number of update steps until a model checkpoint is saved to disk.")
    parser.add_argument('--min_samples_to_start_checkpoints',
                        type=int,
                        default=3000000,
                        help="Number of update steps until model checkpoints start saving to disk.")
    parser.add_argument('--skip_checkpoint',
                        default=False,
                        action='store_true',
                        help="Whether to save checkpoints")
    parser.add_argument('--phase2',
                        default=False,
                        action='store_true',
                        help="Only required for checkpoint saving format")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument('--bert_config_path',
                        type=str,
                        default="/workspace/phase1",
                        help="Path bert_config.json is located in")
    parser.add_argument('--target_mlm_accuracy',
                        type=float,
                        default=0.72,
                        help="Stop training after reaching this Masked-LM accuracy")
    parser.add_argument('--train_mlm_accuracy_window_size',
                        type=int,
                        default=0,
                        help="Average accuracy over this amount of batches before performing a stopping criterion test")
    parser.add_argument('--num_epochs_to_generate_seeds_for',
                        type=int,
                        default=2,
                        help="Number of epochs to plan seeds for. Same set across all workers.")
    parser.add_argument("--use_gradient_as_bucket_view",
                        default=False,
                        action='store_true',
                        help="Turn ON gradient_as_bucket_view optimization in native DDP.")
    parser.add_argument("--dense_seq_output",
                        default=False,
                        action='store_true',
                        help="Whether to run with optimizations.")
    parser.add_argument(
        "--no_ddp", default=False, action="store_true", help="Whether to use DDP."
    )
    parser.add_argument(
        "--dist_lamb",
        default=False,
        action="store_true",
        help="Whether to use DistLamb from pcl_bert",
    )

    # For dtype specific training
    parser.add_argument("--bf16",
                        default=False,
                        action='store_true',
                        help="Enale BFloat16 training")
    parser.add_argument("--bf32",
                        default=False,
                        action='store_true',
                        help="Enale BFloat32 training")
    
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
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
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
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
    parser.add_argument("--local_rank",
                        default=0,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--world_size",
                        default=1,
                        type=int,
                        help="Total batch size for training.")

    parser.add_argument("--profile", action="store_true", help="Whether to enable profiling")
    parser.add_argument("--export_chrome_trace", action="store_true", help="Exports the collected trace in Chrome JSON format.")
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
    parser.add_argument('--dist-url', default='127.0.0.1', type=str,
                     help='url used to set up distributed training')
    parser.add_argument('--dist-port', default='29500', type=str,
                    help='url port used to set up distributed training')
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
    parser.add_argument("--num-iterations", default='10000000000', type=str)
    parser.add_argument("--info", action="store_true")

    args = parser.parse_args()
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    return args


def found_resume_checkpoint(args):
    if args.phase2:
        checkpoint_str = "phase2_ckpt*.pt"
    else:
        checkpoint_str = "phase1_ckpt*.pt"
    return args.resume_from_checkpoint and len(glob.glob(os.path.join(args.output_dir, checkpoint_str))) > 0


def setup_training(args):
    device = torch.device(args.device)
    if int(os.environ.get("PMI_SIZE", "0")) > 1 and not args.multi_instance:
        if args.dist_backend == "ccl":
            try:
                import oneccl_bindings_for_pytorch
            except:
                print("CCL backend requested but import oneccl_bindings_for_pytorch failed")
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

        mpi_world_size = int(os.environ.get('PMI_SIZE', -1))
        if mpi_world_size > 0:
            os.environ['MASTER_ADDR'] = args.dist_url #'127.0.0.1'
            os.environ['MASTER_PORT'] = args.dist_port #'29500'
            os.environ['RANK'] = os.environ.get('PMI_RANK', -1)
            os.environ['WORLD_SIZE'] = os.environ.get('PMI_SIZE', -1)
            args.local_rank = int(os.environ.get('PMI_RANK', -1))
            args.world_size = int(os.environ.get("WORLD_SIZE", -1))
            init_method = 'tcp://' + args.dist_url + ':' + args.dist_port
            torch.distributed.init_process_group(backend=args.dist_backend, init_method=init_method,  world_size=args.world_size, rank=args.local_rank)
            # refer to https://www.open-mpi.org/faq/?category=running#mpi-environmental-variables for MPI ENV  
            _local_rank = os.environ['MPI_LOCALRANKID']
            if 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ.keys():
                _local_rank = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
            device = torch.device("xpu", int(_local_rank))
            print('world_size:{}, rank:{}, device:{}'.format(args.world_size, args.local_rank, device))


        args.local_rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()
        if args.local_rank == 0:
            print(
                f"##################Using {args.dist_backend.upper()} dist run with {args.world_size} ranks",
                flush=True,
            )
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if args.train_batch_size % args.gradient_accumulation_steps != 0:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, batch size {} should be divisible".format(
            args.gradient_accumulation_steps, args.train_batch_size))

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
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    config.dense_seq_output = args.dense_seq_output 
    if args.model_name_or_path:
        model = AutoModelForPreTraining.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForPreTraining.from_config(config)

    param_optimizer = list(model.named_parameters())
    
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay_rate},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    
    print(
        "GroupSizes: ",
        [sum([p.numel() for p in g["params"]]) for g in optimizer_grouped_parameters],
    )
    
    model = model.to(device)
    if torch.distributed.is_initialized() and not args.no_ddp:
        torch.xpu.set_device(device)
        model.xpu(device)
    if args.dist_lamb:
        raise NotImplementedError('Not implemeted for dist_lamb')
    elif args.adamw:
        if args.device == 'xpu':
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=args.learning_rate)
            model, optimizer = torch.xpu.optimize(model=model, optimizer=optimizer, dtype=optimize_dtype)
        elif args.device == 'cuda':
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=args.learning_rate)
    elif args.lamb:
        optimizer = Lamb(
            optimizer_grouped_parameters, 
            lr=args.learning_rate, 
            betas=(args.opt_lamb_beta_1, args.opt_lamb_beta_2))
    else:
        raise NotImplementedError('Please use valid optimizer')

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
        degree=1.0)
    
    if args.device == 'xpu':
        pass
    else:
        if args.bf16 and not args.amp:
            model = model.bfloat16()

    if torch.distributed.is_initialized() and not args.no_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True)
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
        for batch in eval_dataloader:
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
    seq_acc_t = torch.argmax(seq_relationship_logits, dim=-1).eq(next_sentence_label.view([-1])).to(torch.float)
    seq_acc_true, seq_tot = seq_acc_t.sum().item(), seq_acc_t.numel()
    seq_acc = seq_acc_true / seq_tot
    return loss, mlm_acc, num_masked, seq_acc, seq_tot


def main():
    latency_list = []
    args = parse_args()
    status = 'aborted'  # later set to 'success' if termination criteria met
    device, args = setup_training(args)
    print("####################args.world_size={}, args.local_rank={}".format(args.world_size, args.local_rank))
    total_batch_size = global_batch_size(args) 
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
    model, optimizer, lr_scheduler, checkpoint, global_step = prepare_model_and_optimizer(args, device)
    worker_seeds, shuffling_seeds = utils.setup_seeds(args.seed, args.num_epochs_to_generate_seeds_for, device)
    worker_seed = worker_seeds[args.local_rank]
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    worker_init = WorkerInitObj(worker_seed)

    samples_trained = global_step * args.train_batch_size * args.gradient_accumulation_steps * args.world_size
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
        samples_trained_per_step = args.train_batch_size * args.gradient_accumulation_steps * args.world_size
        start, stop, step = args.eval_iter_start_samples, args.max_samples_termination, args.eval_iter_samples
        eval_steps = [math.ceil(i/samples_trained_per_step) for i in np.arange(start, stop, step)]
        eval_count = 0
        next_eval_step = eval_steps[eval_count]

        if args.target_mlm_accuracy:
            if args.train_mlm_accuracy_window_size > 0:
                accuracy_scores = []
                avg_mlm_accuracy = torch.Tensor([0])

        first_epoch = True
        if found_resume_checkpoint(args):
            # dead path
            f_start_id = checkpoint['files'][0]
            files = checkpoint['files'][1:]
            num_files = len(files)
        else:
            files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if
                     os.path.isfile(os.path.join(args.input_dir, f)) and 'part' in f]
            files.sort()
            num_files = len(files)
            random.Random(shuffling_seeds[epoch%len(shuffling_seeds)]).shuffle(files)
            f_start_id = 0
    
    # if torch.distributed.is_initialized() and not args.no_ddp:
    #     model = torch.nn.parallel.DistributedDataParallel(model,
    #                                                     #   find_unused_parameters=True,
    #                                                     #   bucket_cap_mb=8192,
    #                                                       gradient_as_bucket_view=args.use_gradient_as_bucket_view)

    sync_clock()

    while global_step < args.max_steps and not end_training:
        if args.local_rank == 0 or args.local_rank == -1:
            now_time = time.time()
            print("epoch:", epoch)

        # Reshuffle file list on subsequent epochs
        if not first_epoch:
            files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if
                     os.path.isfile(os.path.join(args.input_dir, f)) and 'part' in f]
            files.sort()
            num_files = len(files)
            random.Random(shuffling_seeds[epoch%len(shuffling_seeds)]).shuffle(files)
            f_start_id = 0
        first_epoch = False

        shared_file_list = {}

        if torch.distributed.is_initialized() and args.world_size > num_files:
            remainder = args.world_size % num_files
            data_file = files[(f_start_id*args.world_size + args.local_rank +
                               remainder * f_start_id) % num_files]
        else:
            data_file = files[(f_start_id*args.world_size + args.local_rank) % num_files]
        
        train_data = pretraining_dataset(data_file, args.max_predictions_per_seq)
        if torch.distributed.is_initialized() and not args.no_ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        else:
            train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        
        for f_id in range(f_start_id, len(files)):
            if args.world_size > num_files:
                data_file = files[(f_id*args.world_size + args.local_rank +
                                   remainder * f_id) % num_files]
            else:
                data_file = files[(f_id*args.world_size + args.local_rank)%num_files]

            for step, batch in enumerate(train_dataloader):
                if training_steps >= 10 + int(args.num_iterations):
                    latency_list = latency_list[10:]
                    avg = sum(latency_list) / len(latency_list)
                    print('bert_train latency:  ' + str(avg) + '  s')
                    print('bert_train throughput:  ' + str(total_batch_size / args.world_size / avg) + '  sentences/s')
                    print('perplexity = ' + str(gloss))
                    return
                training_steps += 1

                start_h2d = get_time_sync(args.device)
                
                input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = batch
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
                    "masked_lm_labels": [masked_lm_labels.shape, masked_lm_labels.dtype],
                    "next_sentence_labels": [next_sentence_labels.shape, next_sentence_labels.dtype]
                }
                if args.info:
                    print("datainfo", info)

                def step_training():
                    outputs = None
                    if args.bf16 and args.amp:
                        if args.device == 'cpu':
                            with torch.cpu.amp.autocast():
                                outputs = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                                    labels=masked_lm_labels, next_sentence_label=next_sentence_labels)
                        elif args.device == 'xpu':
                            with torch.xpu.amp.autocast(enabled=True, dtype=optimize_dtype):
                                outputs = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                                    labels=masked_lm_labels, next_sentence_label=next_sentence_labels)
                        elif args.device == 'cuda':
                            with torch.cuda.amp.autocast():
                                outputs = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                                    labels=masked_lm_labels, next_sentence_label=next_sentence_labels)
                    else:
                        outputs = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                                    labels=masked_lm_labels, next_sentence_label=next_sentence_labels)
                    loss = outputs.loss
                    loss = loss / args.gradient_accumulation_steps
                    loss.backward()
                    if hasattr(optimizer, "clip_grad_norm_"):
                        ggnorm = optimizer.clip_grad_norm_(1.0)
                    else:
                        ggnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(
                        train_dataloader
                    ) - 1:
                        start_opt = get_time_sync(args.device)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                        end_opt = get_time_sync(args.device)
                    return outputs, end_opt - start_opt
                
                start_training = get_time_sync(args.device)
                if args.device == 'xpu':
                    with torch.autograd.profiler_legacy.profile(args.profile, use_xpu=True) as prof:
                        outputs, opt_time = step_training()
                    if args.profile:
                        print(str(prof.key_averages().table(sort_by="self_xpu_time_total")))
                        if args.export_chrome_trace and step > 20:
                            with open('./profile_trace.txt', 'w') as f:
                                f.write(str(prof.table(sort_by="id", row_limit=100000)))
                            prof.export_chrome_trace('./profile_trace.json')
                            raise
                elif args.device == 'cuda':
                    if args.profile:
                        with torch.profiler.profile(
                            activities=[
                                # torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ]
                        ) as prof:
                            outputs, opt_time = step_training()
                    else:
                        outputs, opt_time = step_training()
                    if args.profile:
                        print(str(prof.key_averages().table(sort_by="self_cuda_time_total")))
                        if args.export_chrome_trace and step > 20:
                            prof.export_chrome_trace('./profile_trace.json')
                            raise
                end_training = get_time_sync(args.device)
                
                timeinfo = {
                    'h2d': end_h2d - start_h2d,
                    'training': end_training - start_training,
                    'opt': opt_time, # training include opt_time
                }
                timeinfo['total'] = timeinfo['h2d'] + timeinfo['training']
                latency_list.append(timeinfo['total']/1000.0)
                if args.info:
                    print('timeinfo', timeinfo)
                if training_steps > 10:
                    print('train perf:  ' + str(timeinfo['total']/1000.0) + '  s/it')

                gloss, lm_acc, num_masked, seq_acc, seq_tot = calc_accuracy(
                    outputs, masked_lm_labels, next_sentence_labels, args
                )

                if args.benchmark_steps > 0 and global_step + 1 >= args.benchmark_steps:
                    synchronize(args.device)
                    if args.local_rank == 0:
                        print(f"Done Benchmarking {args.benchmark_steps} steps.")
                    sys.exit(0)

                info = {
                    "gloss": gloss,
                    "lm_acc": lm_acc,
                    "num_masked": num_masked,
                    "seq_acc": seq_acc,
                    "seq_tot": seq_tot
                }
                if args.info:
                    print("outinfo", info)

                update_step = training_steps % args.gradient_accumulation_steps == 0
                if update_step:
                    global_step += 1
                    if (
                        args.eval_dir
                        and args.eval_iter_samples > 0
                        and global_step == next_eval_step
                    ):
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
                            print(
                                {
                                    "global_steps": global_step,
                                    "eval_loss": eval_avg_loss,
                                    "eval_mlm_accuracy": eval_avg_mlm_accuracy,
                                }
                            )
                        eval_count += 1
                        next_eval_step = eval_steps[eval_count]

            del train_dataloader
            if samples_trained >= args.max_samples_termination or end_training:
                status = 'success' if converged else 'aborted'
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
    return args, final_loss, train_time_raw


if __name__ == "__main__":
    main()
