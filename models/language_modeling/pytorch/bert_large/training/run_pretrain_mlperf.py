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

import argparse
import logging
import math
import os
import random

import datasets
import torch
from datasets import load_dataset
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
import time

import transformers
from accelerate import Accelerator, DistributedType
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForPreTraining,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    SchedulerType,
    get_scheduler,
    set_seed,
)

from tfrecord.torch.dataset import TFRecordDataset, MultiTFRecordDataset
import intel_extension_for_pytorch as ipex 

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Masked Language Modeling task")
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
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
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
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )

    parser.add_argument("--profile", action="store_true", help="Whether to enable profiling")

    parser.add_argument("--benchmark", action="store_true", help="Whether to enable benchmark")
    parser.add_argument("--bf16", action="store_true", help="Whether to enable benchmark")

    args = parser.parse_args()

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


def main():
    args = parse_args()
    print(args)
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator(split_batches=False)
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        #transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        #transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Load pretrained model
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.model_name_or_path:
        model = AutoModelForPreTraining.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForPreTraining.from_config(config)

    print(model)
    if args.max_seq_length is not None and args.max_seq_length != 512:
        raise ValueError(f"Max_seq_length is fixed to 512 for MLPerf dataset but args.max_seq_length = {args.max_seq_length} specified")

    def transform_features(features):
        features['attention_mask'] = features['input_mask']
        del features['input_mask']
        features['token_type_ids'] = features['segment_ids'].to(torch.int64)
        del features['segment_ids']
        features['next_sentence_label'] = features['next_sentence_labels'].to(torch.int64)
        del features['next_sentence_labels']
        labels = torch.ones_like(features['input_ids']) * -100
        # FIXME: use features['masked_lm_weights']
        nlabels = labels.scatter(-1, features['masked_lm_positions'].to(torch.int64), features['masked_lm_ids'])
        # Fix for padded lm_positions
        nlabels[:,0] = labels[:,0]
        labels = nlabels
        features['labels'] = labels.to(torch.int64)
        features['input_ids'] = features['input_ids'].to(torch.int64)
        del features['masked_lm_positions']
        del features['masked_lm_ids']
        #print(f"masked_lm_weights count = {features['masked_lm_weights'].sum()}")
        del features['masked_lm_weights']
        return features

    train_dataset = TFRecordDataset(args.train_file, index_path=None, description=None) #, transform=transform_features)
    eval_dataset = TFRecordDataset(args.validation_file, index_path=None, description=None) #, transform=transform_features)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.per_device_eval_batch_size)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.per_device_train_batch_size)
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare everything with our `accelerator`.
    #model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    #    model, optimizer, train_dataloader, eval_dataloader
    #)

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    epoch_len = 0
    try:
        epoch_len = len(train_dataloader)
    except:
        # for _ in train_dataloader: epoch_len += 1
        pass

    num_update_steps_per_epoch = math.ceil(epoch_len / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        if num_update_steps_per_epoch > 0:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        else:
            args.max_train_steps = 100000000 #some large number
    else:
        if num_update_steps_per_epoch > 0:
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    def trace_handler(prof):
        print(prof.key_averages().table(
            sort_by="self_cpu_time_total", row_limit=-1))
        prof.export_chrome_trace("./log/test_trace_" + str(prof.step_num) + ".json")

    def calc_accuracy(outputs, masked_lm_labels, next_sentence_label, sync_global = True):
        #for k in outputs: print(k)
        prediction_logits = outputs.prediction_logits
        seq_relationship_logits = outputs.seq_relationship_logits
        lm_acc_t = torch.masked_select(torch.argmax(prediction_logits, dim=-1).eq(masked_lm_labels).to(torch.float), masked_lm_labels.ne(-100))
        seq_acc_t = torch.argmax(seq_relationship_logits, dim=-1).eq(next_sentence_label.view([-1])).to(torch.float)
        metrics = [outputs.loss.item(), lm_acc_t.sum().item(), lm_acc_t.numel(), seq_acc_t.sum().item(), seq_acc_t.numel()]
        if sync_global == True:
            tmp = [metrics]
            tmp = torch.tensor(tmp, dtype=torch.float)
            tmp = accelerator.gather(tmp).sum(0)
            metrics = [i.item() for i in tmp]
            metrics[0] = metrics[0] / accelerator.num_processes
        (loss, lm_true, lm_tot, seq_true, seq_tot) = metrics
        lm_acc = lm_true / lm_tot
        seq_acc = seq_true / seq_tot
        #if accelerator.is_main_process:
        #    print(f"loss = {loss:.3f}  lm_acc = {lm_acc:.3f} ({lm_true:.0f}/{lm_tot:.0f}),  seq_acc = {seq_acc:.3f} ({seq_true:.0f}/{seq_tot:.0f})")
        return loss, lm_acc, seq_acc, lm_true, lm_tot, seq_true, seq_tot
     
    def train(batch, model, optmizer, lr_scheduler, step):       
        t1 = time.time()
        batch = transform_features(batch)
        #print(f"Input shape: {batch['input_ids'].shape}")
        t2 = time.time()
        outputs = model(**batch)
        t3 = time.time()
        loss = outputs.loss
        loss = loss / args.gradient_accumulation_steps
        loss.backward()
        t4 = time.time()
        if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            #progress_bar.update(1)
        t5 = time.time()
        #prof1.step()
        gloss, lm_acc, seq_acc, lm_true, lm_tot, seq_true, seq_tot = calc_accuracy(outputs, batch['labels'], batch['next_sentence_label'])
        if accelerator.is_main_process:
            print(f"Step {completed_steps:5d}: loss: {gloss:6.3f} lm_acc: {lm_acc:.3f} seq_acc: {seq_acc:.3f} lbs: {args.per_device_train_batch_size} gbs: {total_batch_size} XT: {(t2-t1)*1000.0:.1f} FT: {(t3-t2)*1000.0:.1f} BT: {(t4-t3)*1000.0:.1f} OT: {(t5-t4)*1000.0:.1f} TT: {(t5-t1)*1000.0:.1f}", flush=True)
        
    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    #logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    #progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    t0 = time.time()
    bench_total_time = 0
    model.train()
    model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.bfloat16 if args.bf16 else torch.float)
    #with torch.profiler.profile(
    #    activities=[
    #        torch.profiler.ProfilerActivity.CPU],

    #    schedule=torch.profiler.schedule(
    #        wait=1,
    #        warmup=9,
    #        active=10),
    #    #on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/bert_bf16'),#trace_handler
    #    on_trace_ready=trace_handler#torch.profiler.tensorboard_trace_handler('./log/bert_bf16')
    #    # used when outputting for tensorboard
    #    ) as prof1:

    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            record_shapes = True
            t_beg = time.time()
            if args.bf16:
                with torch.cpu.amp.autocast():
                    train(batch, model, optimizer, lr_scheduler, step)
            else:
                train(batch, model, optimizer, lr_scheduler, step)
            t_end = time.time()
            completed_steps += 1
            if completed_steps >= args.max_train_steps:
                break
            if args.benchmark and completed_steps > 10:
                bench_total_time = bench_total_time + (t_end -t_beg)
            if args.benchmark and completed_steps > 110:
                throughput = 100 * args.per_device_train_batch_size / bench_total_time
                print("Throughput: {:.3f} sentence/s".format(throughput), flush=True)
                if args.output_dir is not None:
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
                exit(0)

        model.eval()
        losses = []
        lm_trues = 0
        lm_total = 0
        seq_trues = 0
        seq_total = 0
        for step, batch in enumerate(eval_dataloader):
            batch = transform_features(batch)
            with torch.cpu.amp.autocast(), torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            gloss, lm_acc, seq_acc, lm_true, lm_tot, seq_true, seq_tot = calc_accuracy(outputs, batch['labels'], batch['next_sentence_label'])
            losses.append(gloss)
            lm_trues += lm_true
            lm_total += lm_tot
            seq_trues += seq_true
            seq_total += seq_tot
            if accelerator.is_main_process:
                print(f"Eval Step: {step}  loss: {gloss:.3f} lm_acc: {lm_acc:.3f}  seq_acc: {seq_acc:.3f} lbs: {args.per_device_eval_batch_size} gbs: {args.per_device_eval_batch_size*accelerator.num_processes}", flush=True)

            if step*args.per_device_eval_batch_size*accelerator.num_processes > 10000: break

        g_loss = sum(losses) / len(losses)
        g_lm_acc = lm_trues / lm_total
        g_seq_acc = seq_trues / seq_total
        if accelerator.is_main_process:
            print(f"Global Eval loss: {g_loss:.3f} lm_acc: {g_lm_acc:.3f}  seq_acc: {g_seq_acc:.3f}", flush=True)

        #logger.info(f"epoch {epoch}: perplexity: {perplexity}")

        if completed_steps >= args.max_train_steps:
            break
    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)


if __name__ == "__main__":
    main()
