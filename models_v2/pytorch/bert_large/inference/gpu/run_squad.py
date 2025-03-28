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
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""


import argparse
import glob
import logging
import os
import random
import timeit
import time
import contextlib

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

import transformers
from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)

from transformers.modeling_outputs import (
    QuestionAnsweringModelOutput,
)

from transformers.data.processors.squad import (
    SquadResult,
    SquadV1Processor,
    SquadV2Processor,
)
from transformers.trainer_utils import is_main_process


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

MAP_TORCH_DTYPE = {"FP32": torch.float32, "FP16": torch.float16,
                   "BF16": torch.bfloat16, "FP64": torch.float64,
                   "CF64": torch.cfloat, "CF128": torch.cdouble,
                   "UINT8": torch.uint8, "INT8": torch.int8,
                   "INT16": torch.int16, "INT": torch.int32,
                   "INT64": torch.int64, "BOOL": torch.bool,
                  }

hub = os.path.expanduser("~/.cache/torch/intel")
if not os.path.exists(hub):
    os.makedirs(hub)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_pcl:
        pcl_bert.set_rnd_seed(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def load_model(args, config = None, checkpoint = None):
    model = None
    if checkpoint and args.use_pcl:
        with pcl_bert.pcl_impl(args.use_pcl, args.pcl_bf16, args.unpad):
            model = AutoModelForQuestionAnswering.from_pretrained(
                checkpoint
            )  # , force_download=True)

    elif checkpoint and (not args.use_pcl):
        model = AutoModelForQuestionAnswering.from_pretrained(
            checkpoint
        )  # , force_download=True)

    elif (not checkpoint) and args.use_pcl:
        with pcl_bert.pcl_impl(args.use_pcl, args.pcl_bf16, args.unpad):
            model = AutoModelForQuestionAnswering.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )
    else:
        model = AutoModelForQuestionAnswering.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
    return model

def load_jit_model(model, inputs, dtype, device, jit_trace_path, use_jit_cache):
    jit_model = None
    if (os.path.isfile(jit_trace_path)) and use_jit_cache:
        print("load trace model ...")
        jit_model = torch.jit.load(jit_trace_path)
        print("load trace model done")
    else:
        print("create trace model")
        in_1 = torch.unsqueeze(inputs["input_ids"][0].clone(), 0)
        in_2 = torch.unsqueeze(inputs["token_type_ids"][0].clone(), 0)
        in_3 = torch.unsqueeze(inputs["attention_mask"][0].clone(), 0)
        with torch.autocast("xpu", enabled=True, dtype=dtype, cache_enabled=False):
            jit_model = torch.jit.trace(model,
                (in_1.to(device),
                in_2.to(device),
                in_3.to(device)),
                strict = False)
        jit_model.save(jit_trace_path)

    return jit_model

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def collate_fn_(batch, device=None, dtype=None):
    for key, value in batch.items():
        if device:
            batch[key] = value.to(device)
        if isinstance(value, torch.Tensor) and value.dtype == torch.float and dtype is not None and dtype != "FP32":
            batch[key] = value.to(MAP_TORCH_DTYPE[dtype])
    return batch

def model_cast(model, device=None, dtype=None):
    # Set device type
    if (device):
        model.to(device)

    # Set data type
    if (dtype == None):
        pass
    elif (dtype == "FP16"):
        model.half()
    elif (dtype == "BF16"):
        model.bfloat16()
    elif (dtype == "FP64"):
        model.double()
    elif (dtype == "FP32"):
        model.float()
    else:
        logger.error("The datatype for model casting not yet supported by pytorch")

def get_device(device_choice):
    if device_choice == "xpu":
        device = torch.device(
            "xpu" if torch.xpu.is_available() else "cpu"
        )
        n_gpu = torch.xpu.device_count() if torch.xpu.is_available() else 0
    elif device_choice == "cuda":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    else :
        device = torch.device("cpu")
        n_gpu = 0
    return device, n_gpu


def train(args, train_dataset, model, tokenizer):
    """Train the model"""
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = (
        RandomSampler(train_dataset)
        if args.local_rank == -1
        else DistributedSampler(train_dataset)
    )
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (
            args.max_steps
            // (len(train_dataloader) // args.gradient_accumulation_steps)
            + 1
        )
    else:
        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    if args.use_pcl:
        optimizer = pcl_bert.AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
        )
    else:
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
        )

    if args.device_choice == 'xpu' and args.optimize:
        optimize_dtype = torch.float32
        if args.dtype == "FP16":
            optimize_dtype = torch.float16
        elif args.dtype == "BF16":
            optimize_dtype = torch.bfloat16
        model, optimizer = torch.xpu.optimize(model=model, optimizer=optimizer, level="O1",
                                              dtype=optimize_dtype)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(
        os.path.join(args.model_name_or_path, "optimizer.pt")
    ) and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "optimizer.pt"))
        )
        scheduler.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "scheduler.pt"))
        )

    if (args.dtype == "FP16") and (args.device_choice == "cuda"):
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )

        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level
        )

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        if args.n_gpu == 0:
            model = torch.nn.parallel.DistributedDataParallel(
                model, find_unused_parameters=True
            )
        else:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True,
            )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info(
        "  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size
    )
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (
                len(train_dataloader) // args.gradient_accumulation_steps
            )
            steps_trained_in_current_epoch = global_step % (
                len(train_dataloader) // args.gradient_accumulation_steps
            )

            logger.info(
                "  Continuing training from checkpoint, will skip to saved global_step"
            )
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info(
                "  Will skip the first %d steps in the first epoch",
                steps_trained_in_current_epoch,
            )
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained,
        int(args.num_train_epochs),
        desc="Epoch",
        disable=args.local_rank not in [-1, 0],
    )
    # Added here for reproductibility
    set_seed(args)

    for _ in train_iterator:
        epoch_iterator = tqdm(
            train_dataloader, desc="Iteration", disable=True
        )  # args.local_rank not in [-1, 0])
        # epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        end_time = timeit.default_timer()

        # for perf compute
        time_collect = []

        # autocast dtype
        use_autocast = False
        autocast_dtype = torch.float32
        if args.dtype == "FP16":
            use_autocast = True
            autocast_dtype = torch.float16
        elif args.dtype == "BF16":
            use_autocast = True
            autocast_dtype = torch.bfloat16

        # autocast context
        if args.device_choice == 'cuda':
            autocast_context = torch.autocast("cuda", enabled=use_autocast, dtype=autocast_dtype)
        elif args.device_choice == 'xpu':
            autocast_context = torch.autocast("xpu", enabled=use_autocast, dtype=autocast_dtype)
        else:
            autocast_context = torch.autocast("cpu", enabled=use_autocast, dtype=autocast_dtype)

        import contextlib
        profile_context = contextlib.nullcontext()

        def profiler_setup(mode="disable", legacy_profilers=[], **prof_kwargs):
            # initial for clear stage
            legacy_profilers.clear()
            num_iters = args.num_steps if args.num_steps else len(epoch_iterator)
            if mode == "disable":
                legacy_profilers.extend(
                    [contextlib.nullcontext() for _ in range(num_iters)]
                )
                return contextlib.nullcontext()
            if mode == "legacy":
                legacy_kwargs = {"enabled": True}
                if args.device_choice:
                    legacy_kwargs[f"use_{args.device_choice}"] = True
                legacy_profilers.extend(
                    [torch.autograd.profiler_legacy.profile(**legacy_kwargs, **prof_kwargs)
                     for _ in range(num_iters)]
                )
                return contextlib.nullcontext()
            elif mode == "kineto":
                legacy_profilers.extend(
                    [contextlib.nullcontext() for _ in range(num_iters)]
                )
                # configure for kineto profiler
                activities = [torch.profiler.ProfilerActivity.CPU]
                if args.device_choice == 'cuda':
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                elif args.device_choice == 'xpu':
                    activities.append(torch.profiler.ProfilerActivity.XPU)
                skip_iters = max(num_iters - 5, 0)
                my_schedule = torch.profiler.schedule(skip_first=skip_iters,
                                                      wait=1, warmup=3, active=1)
                def trace_handle(prof):
                    profile_name = 'fp32'
                    if args.dtype == "FP16":
                        profile_name = 'fp16'
                    elif args.dtype == "BF16":
                        profile_name = 'bf16'
                    if args.device_choice == 'xpu':
                        torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"), './profiling.' + profile_name + '.train.pt')
                    elif args.device_choice == 'cuda':
                        torch.save(prof.key_averages().table(sort_by="self_cuda_time_total"), './profiling.card.' + str(args.xpu) + '.pt')
                    else:
                        torch.save(prof.key_averages().table(sort_by="self_cpu_time_total"), './profiling.card.' + str(args.xpu) + '.pt')
                return torch.profiler.profile(activities=activities, schedule=my_schedule, on_trace_ready=trace_handle, **prof_kwargs)

        def legacy_profile_print(prof, sort_item: str):
            print(prof.key_averages().table(sort_by=sort_item))
            print(prof.key_averages(group_by_input_shape=True).table())


        def train_iter(step, batch):
            # Skip past any already trained steps if resuming training
            try:
                import memory_check
                memory_check.display_mem("xpu:0")
            except:
                pass

            nonlocal steps_trained_in_current_epoch
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                return

            start_fwd_time = timeit.default_timer()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }

            if args.model_type in [
                "xlm",
                "roberta",
                "distilbert",
                "camembert",
                "bart",
                "longformer",
            ]:
                del inputs["token_type_ids"]

            if args.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[5], "p_mask": batch[6]})
                if args.version_2_with_negative:
                    inputs.update({"is_impossible": batch[7]})
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {
                            "langs": (
                                torch.ones(batch[0].shape, dtype=torch.int64)
                                * args.lang_id
                            ).to(args.device)
                        }
                    )

            with autocast_context:
                outputs = model(**inputs)

            loss = outputs[0]

            if args.n_gpu > 1:
                loss = (
                    loss.mean()
                )  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            start_bwd_time = timeit.default_timer()
            if args.dtype == "FP16" and args.device_choice == "cuda":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            start_opt_time = timeit.default_timer()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                with torch.autograd.profiler.record_function("clip_grad_norm"):
                    if args.dtype == "FP16":
                        torch.nn.utils.clip_grad_norm_(
                            amp.master_params(optimizer), args.max_grad_norm
                        )
                    elif args.use_pcl and args.pcl_bf16:
                        pcl_bert.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm
                        )
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm
                        )

                with torch.autograd.profiler.record_function("optimizer"):
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    # model.zero_grad()
                    for p in model.parameters():
                        p.grad = None
                nonlocal global_step
                global_step += 1

                step_loss = loss.item()
                nonlocal tr_loss
                tr_loss += step_loss

                nonlocal end_time
                data_time = start_fwd_time - end_time
                end_time = timeit.default_timer()

                nonlocal time_collect
                # kick off the first step
                if global_step >= 1:
                    time_collect.append(end_time - start_fwd_time + data_time) # unit: s

                # Log metrics
                if (
                    args.local_rank in [-1, 0]
                    and args.logging_steps > 0
                    and global_step % args.logging_steps == 0
                ):
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar(
                                "eval_{}".format(key), value, global_step
                            )
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar(
                        "loss",
                        (tr_loss - logging_loss) / args.logging_steps,
                        global_step,
                    )
                    logging_loss = tr_loss

                # Save model checkpoint
                if (
                    args.local_rank in [-1, 0]
                    and args.save_steps > 0
                    and global_step % args.save_steps == 0
                ):
                    output_dir = os.path.join(
                        args.output_dir, "checkpoint-{}".format(global_step)
                    )
                    # Take care of distributed/parallel training
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(
                        optimizer.state_dict(),
                        os.path.join(output_dir, "optimizer.pt"),
                    )
                    torch.save(
                        scheduler.state_dict(),
                        os.path.join(output_dir, "scheduler.pt"),
                    )
                    logger.info(
                        "Saving optimizer and scheduler states to %s", output_dir
                    )

            if args.local_rank in [-1, 0]:
                print(
                    f"Step: {global_step-1}, loss: {step_loss:6g}  tr_loss: {tr_loss/(global_step-1):6g} DT: {data_time*1e3:6g} FT: {(start_bwd_time-start_fwd_time)*1e3:6g} BT: {(start_opt_time-start_bwd_time)*1e3:6g} OT: {(end_time-start_opt_time)*1e3:6g} TT: {(end_time-start_fwd_time+data_time)*1e3:6g}"
                )
            return tr_loss

        num_step = 0
        mode = "kineto" if args.kineto_profile else "legacy" if args.legacy_profile else "disable"
        legacy_profilers = []
        with profiler_setup(mode, legacy_profilers, record_shapes=True) as prof:
            for step, batch in enumerate(epoch_iterator):
                with legacy_profilers[step] as prof_legacy:
                    loss = train_iter(step, batch)
                num_step += 1
                if mode == "kineto":
                    prof.step()
                elif mode == "legacy":
                    legacy_profile_print(prof_legacy, f"self_{args.device_choice}_time_total")

                if args.num_steps != -1 and num_step >= args.num_steps:
                    time_collect_num = len(time_collect)
                    avg_time = sum(time_collect) / time_collect_num
                    print("train avg loss is:", loss)
                    print("train latency is:", avg_time, ' s')
                    print("train throughput is:", args.train_batch_size/avg_time, ' sentences/s')
                    import sys
                    sys.exit()

                if args.max_steps > 0 and global_step > args.max_steps:
                    epoch_iterator.close()
                    return

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    model_cast(model, device=args.device)
    model.eval()

    dataset, examples, features = load_and_cache_examples(
        args, tokenizer, evaluate=True, output_examples=True
    )

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(
        dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
    )

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_results = []

    jit_model = None
    jit_trace_path = os.path.join(
        hub,
        args.model_type + "_" +
        args.model_name_or_path.rsplit("/", 1)[-1] + "_trace_" + 
        args.dtype + ".zip"
    )

    do_profiling = os.environ.get("PROFILE", "OFF").upper() in ["1", "Y", "ON", "YES", "TRUE"]

    def inference_profiler_setup(args, mode, **prof_kwargs):
        if mode == "legacy":
            logger.warning("Legacy profiler has been depracated")
            return contextlib.nullcontext()
        elif mode == "kineto":
            activities = [torch.profiler.ProfilerActivity.CPU]
            if args.device_choice == 'cuda':
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            elif args.device_choice == 'xpu':
                activities.append(torch.profiler.ProfilerActivity.XPU)

            def trace_handle(prof):
                profile_name = 'fp32'
                if args.dtype == "FP16":
                    profile_name = 'fp16'
                elif args.dtype == "BF16":
                    profile_name = 'bf16'
                profile_dir = './bert-inference-profile'
                if os.path.exists(profile_dir):
                    if not os.path.isdir(profile_dir):
                        profile_dir = './'
                else:
                    os.makedirs(profile_dir, exist_ok=True)
                if args.device_choice == 'xpu':
                    torch.save(
                        prof.key_averages().table(sort_by="self_xpu_time_total"),
                        f'{profile_dir}/inference-profiling.xpu.{profile_name}.step{prof.step_num}.pt'
                    )
                elif args.device_choice == 'cuda':
                    torch.save(
                        prof.key_averages().table(sort_by="self_cuda_time_total"),
                        f'{profile_dir}/inference-profiling.cuda.{profile_name}.step{prof.step_num}.pt'
                    )
                else:
                    torch.save(
                        prof.key_averages().table(sort_by="self_cpu_time_total"),
                        f'{profile_dir}/inference-profiling.cpu.{profile_name}.step{prof.step_num}.pt'
                    )
            return torch.profiler.profile(
                activities=activities,
                schedule=torch.profiler.schedule(
                    skip_first=0,
                    wait=6,
                    warmup=3,
                    active=1,
                ),
                on_trace_ready=trace_handle,
                **prof_kwargs,
            )
        return contextlib.nullcontext()

    start_time = time.time()
    num_step = 0
    time_collect = []
    mode = "kineto" if args.kineto_profile else "legacy" if args.legacy_profile else "disable"
    with inference_profiler_setup(args, mode, record_shapes=False) as prof:
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            if args.model_type in [
                "xlm",
                "roberta",
                "distilbert",
                "camembert",
                "bart",
                "longformer",
            ]:
                del inputs["token_type_ids"]

            feature_indices = batch[3]

            # XLNet and XLM use more arguments for their predictions
            if args.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
                # for lang_id-sensitive xlm models
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {
                            "langs": (
                                torch.ones(batch[0].shape, dtype=torch.int64)
                                * args.lang_id
                            ).to(args.device)
                        }
                    )

            if num_step == 0:
                # optimize model
                model = torch.xpu.optimize(
                    model=model,
                    dtype=MAP_TORCH_DTYPE[args.dtype],
                    level="O1",
                    weights_prepack=False,
                )
                if args.do_jit:
                    jit_model = load_jit_model(
                        model,
                        inputs,
                        MAP_TORCH_DTYPE[args.dtype],
                        args.device,
                        jit_trace_path,
                        args.jit_cache,
                    )
                if args.do_dynamo:
                    dynamo_model = torch.compile(model)
                    sample_inputs = collate_fn_(inputs, device=args.device)
                    _ = dynamo_model(**sample_inputs)

            with torch.inference_mode():
                outputs = None
                batch_start = None
                batch_end = None
                with torch.autograd.profiler_legacy.profile(
                    do_profiling, use_xpu=True, record_shapes=False
                ) if mode == 'legacy' else contextlib.nullcontext() as legacy_prof:
                    batch_start = time.time()
                    inputs = collate_fn_(inputs, device=args.device)
                    if args.do_jit:
                        outputs = jit_model(**inputs)
                    elif args.do_dynamo:
                        outputs = dynamo_model(**inputs)
                    else:
                        outputs = model(**inputs)
                    for _, v in outputs.items():
                        v = v.to(torch.float32).to("cpu")
                    batch_end = time.time()
                print("local latency is:", (batch_end - batch_start), ' s')
                print("local throughput is:", args.eval_batch_size/(batch_end - batch_start), ' sentences/s')

                if num_step >= 10 and num_step <= (len(dataset)/args.eval_batch_size - 10):
                    time_collect.append(batch_end - batch_start)

                if args.do_jit:
                    outputs = QuestionAnsweringModelOutput(
                        start_logits=outputs["start_logits"],
                        end_logits=outputs["end_logits"],
                    )

                if num_step == 10 and do_profiling and mode == 'legacy':
                    profiling_path = os.getenv('PROFILE_PATH')
                    # if no set PROFILE_PATH, use current dir
                    if profiling_path is None:
                        profiling_path = './'
                    if (args.profiling_sub_file != ""):
                        profiling_path += args.profiling_sub_file
                    mkdir(profiling_path)
                    torch.save(legacy_prof.key_averages().table(sort_by="self_xpu_time_total"),
                               profiling_path + '/bert_inference_profile.pt')
                    print(legacy_prof.key_averages().table(sort_by="self_xpu_time_total"))
                    torch.save(legacy_prof.table(sort_by="id", row_limit=100000),
                               profiling_path + '/bert_inference_profile_detailed.pt')
                    legacy_prof.export_chrome_trace(profiling_path + '/bert_inference_profile.json')

            for i, feature_index in enumerate(feature_indices):
                eval_feature = features[feature_index.item()]
                unique_id = int(eval_feature.unique_id)

                output = [to_list(output[i]) for output in outputs.to_tuple()]

                # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
                # models only use two.
                if len(output) >= 5:
                    start_logits = output[0]
                    start_top_index = output[1]
                    end_logits = output[2]
                    end_top_index = output[3]
                    cls_logits = output[4]

                    result = SquadResult(
                        unique_id,
                        start_logits,
                        end_logits,
                        start_top_index=start_top_index,
                        end_top_index=end_top_index,
                        cls_logits=cls_logits,
                    )

                else:
                    start_logits, end_logits = output
                    result = SquadResult(unique_id, start_logits, end_logits)

                all_results.append(result)
            num_step += 1
            if mode == "kineto":
                prof.step()
            if args.num_steps != -1 and num_step >= args.num_steps:
                break

    evalTime = time.time() - start_time

    if do_profiling and mode == 'kineto':
        profiling_path = os.getenv('PROFILE_PATH')
        # if no set PROFILE_PATH, use current dir
        if profiling_path is None:
            profiling_path = './'
        if (args.profiling_sub_file != ""):
            profiling_path += args.profiling_sub_file
        mkdir(profiling_path)
        torch.save(
            prof.key_averages().table(sort_by="self_xpu_time_total"),
            profiling_path + '/bert_inference_profile.pt',
        )
        print(prof.key_averages().table(sort_by="self_xpu_time_total"))
        prof.export_chrome_trace(profiling_path + '/bert_inference_profile.json')

    if len(time_collect) > 0:
        avg_time = sum(time_collect)/len(time_collect)
        print("inf latency: ", avg_time, ' s')
        print("inf throughput: ", args.eval_batch_size/avg_time, ' sentences/s')
    if args.num_steps != -1 and args.num_steps < (len(dataset)/args.eval_batch_size + 1):
        exit()
    logger.info(
        "  Evaluation done in total %f secs (%f sec per example)",
        evalTime,
        evalTime / len(dataset),
    )

    # Compute predictions
    output_prediction_file = os.path.join(
        args.output_dir, "predictions_{}.json".format(prefix)
    )
    output_nbest_file = os.path.join(
        args.output_dir, "nbest_predictions_{}.json".format(prefix)
    )

    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(
            args.output_dir, "null_odds_{}.json".format(prefix)
        )
    else:
        output_null_log_odds_file = None

    # XLNet and XLM use a more complex post-processing procedure
    if args.model_type in ["xlnet", "xlm"]:
        start_n_top = (
            model.config.start_n_top
            if hasattr(model, "config")
            else model.module.config.start_n_top
        )
        end_n_top = (
            model.config.end_n_top
            if hasattr(model, "config")
            else model.module.config.end_n_top
        )

        predictions = compute_predictions_log_probs(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            start_n_top,
            end_n_top,
            args.version_2_with_negative,
            tokenizer,
            args.verbose_logging,
        )
    else:
        predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            args.do_lower_case,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            args.verbose_logging,
            args.version_2_with_negative,
            args.null_score_diff_threshold,
            tokenizer,
        )

    # Compute the F1 and exact scores.
    results = squad_evaluate(examples, predictions)
    return results


def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    input_dir = args.data_dir if args.data_dir else "."
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
        ),
    )

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        logger.info("Creating features from dataset file at %s", input_dir)

        if not args.data_dir and (
            (evaluate and not args.predict_file)
            or (not evaluate and not args.train_file)
        ):
            try:
                import tensorflow_datasets as tfds
            except ImportError:
                raise ImportError(
                    "If not data_dir is specified, tensorflow_datasets needs to be installed."
                )

            if args.version_2_with_negative:
                logger.warn("tensorflow_datasets does not handle version 2 of SQuAD.")

            tfds_examples = tfds.load("squad")
            examples = SquadV1Processor().get_examples_from_dataset(
                tfds_examples, evaluate=evaluate
            )
        else:
            processor = (
                SquadV2Processor()
                if args.version_2_with_negative
                else SquadV1Processor()
            )
            if evaluate:
                examples = processor.get_dev_examples(
                    args.data_dir, filename=args.predict_file
                )
            else:
                examples = processor.get_train_examples(
                    args.data_dir, filename=args.train_file
                )

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
            return_dataset="pt",
            threads=args.threads,
        )

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(
                {"features": features, "dataset": dataset, "examples": examples},
                cached_features_file,
            )

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    if output_examples:
        return dataset, examples, features
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_TYPES),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the .json files for the task."
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from huggingface.co",
    )

    parser.add_argument(
        "--dtype",
        default="FP32",
        type=str,
        choices=["FP32", "BF16", "FP16"],
        help= "Specify precision to use",
    )

    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, the SQuAD examples contain some that do not have an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
    )

    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
        "be truncated to this length.",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--do_jit", action="store_true", help="Whether to run eval with jit on the dev set."
    )
    parser.add_argument(
        "--do_dynamo", action="store_true", help="Whether to run eval with torch.compile on the dev set."
    )
    parser.add_argument(
        "--jit_cache",
        action="store_true",
        default=True,
        help="Whether to use jit model cache",
    )
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=-1,
        help="Number of steps to run model.",
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=3.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
    )
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another.",
    )
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal SQuAD evaluation.",
    )
    parser.add_argument(
        "--lang_id",
        default=0,
        type=int,
        help="language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)",
    )

    parser.add_argument(
        "--logging_steps", type=int, default=500, help="Log every X updates steps."
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument(
        "--device_choice",
        default="cpu",
        type=str,
        choices=["cpu", "xpu", "cuda"],
        help= "Specify device to use when available",
    )
    parser.add_argument(
        "--legacy_profile",
        action="store_true",
        help="Whether to running to get profile data with legacy profiler",
    )
    parser.add_argument(
        "--kineto_profile",
        action="store_true",
        help="Whether to running to get profile data with kineto profiler",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Whether to use ipex.optimize for xpu model",
    )
    parser.add_argument(
        "--use_pcl",
        action="store_true",
        help="Whether to use PCL Fused impl when available",
    )
    parser.add_argument(
        "--pcl_bf16",
        action="store_true",
        help="Whether to use PCL Fused impl when available",
    )
    parser.add_argument(
        "--unpad",
        action="store_true",
        help="Whether to use PCL Fused impl when available",
    )
    parser.add_argument(
        "--dist_backend",
        type=str,
        default="ccl",
        help="Specify distributed backend to use.",
    )
    parser.add_argument(
        "--profiling_sub_file",
        type=str,
        default="",
        help="Path to saved torch profiler",
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )

    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--server_ip", type=str, default="", help="Can be used for distant debugging."
    )
    parser.add_argument(
        "--server_port", type=str, default="", help="Can be used for distant debugging."
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="multiple threads for converting example to features",
    )
    args = parser.parse_args()

    if args.device_choice == 'xpu':
        import intel_extension_for_pytorch

    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    if args.use_pcl:
        try:
            from pcl_pytorch_extension import bert as pcl_bert
        except:
            print("CCL backend requested but import torch_ccl failed")
            raise
    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(
            address=(args.server_ip, args.server_port), redirect_output=True
        )
        ptvsd.wait_for_attach()

    # Setup CUDA, XPU & distributed training
    if int(os.environ.get("PMI_SIZE", "0")) > 1:
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

        os.environ["RANK"] = os.environ.get("PMI_RANK", "0")
        os.environ["WORLD_SIZE"] = os.environ.get("PMI_SIZE", "1")
        torch.distributed.init_process_group(backend=args.dist_backend)
        device = torch.device("cpu")
        args.n_gpu = 0
        args.local_rank = torch.distributed.get_rank()
        print(
            f"Using {args.dist_backend.upper()} dist run with {torch.distributed.get_world_size()} ranks"
        )
    elif args.local_rank == -1:
        device, args.n_gpu = get_device(args.device_choice)
        if args.n_gpu > 1:
            args.n_gpu = 1
    else:
        if args.device_choice == 'cuda':
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            torch.distributed.init_process_group(backend="nccl")
        elif args.device_choice == 'xpu':
            torch.xpu.set_device(args.local_rank)
            device = torch.device("xpu", args.local_rank)
        args.n_gpu = 1
    args.device = device

    if args.legacy_profile or args.kineto_profile:
        os.environ["PROFILE"] = "1"

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, dtype: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.dtype,
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
        use_fast=False,  # SquadDataset is not compatible with Fast tokenizers which have a smarter overflow handeling
    )
    model = load_model(args, config = config)

    for m in model.modules():
        if hasattr(m, "maybe_block_params"):
            m.maybe_block_params()
            # if args.pcl_bf16: m.to(torch.bfloat16)

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.dtype is float16.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if (args.dtype == "FP16") and (args.device_choice == "cuda"):
        try:
            import apex

            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(
            args, tokenizer, evaluate=False, output_examples=False
        )
        model.train()
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Save the trained model and the tokenizer
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = AutoModelForQuestionAnswering.from_pretrained(
            args.output_dir
        )  # , force_download=True)

        # SquadDataset is not compatible with Fast tokenizers which have a smarter overflow handeling
        # So we use use_fast=False here for now until Fast-tokenizer-compatible-examples are out
        tokenizer = AutoTokenizer.from_pretrained(
            args.output_dir, do_lower_case=args.do_lower_case, use_fast=False
        )
        model.to(args.device)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        if args.do_train:
            logger.info("Loading checkpoints saved during training for evaluation")
            checkpoints = [args.output_dir]
            if args.eval_all_checkpoints:
                checkpoints = list(
                    os.path.dirname(c)
                    for c in sorted(
                        glob.glob(
                            args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True
                        )
                    )
                )

        else:
            logger.info("Loading checkpoint %s for evaluation", args.model_name_or_path)
            checkpoints = [args.model_name_or_path]

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = load_model(args, checkpoint = checkpoint)

            # Evaluate
            result = evaluate(args, model, tokenizer, prefix=global_step)

            result = dict(
                (k + ("_{}".format(global_step) if global_step else ""), v)
                for k, v in result.items()
            )
            results.update(result)

    # logger.info("Results: {}".format(results))
    print("Results: {}".format(results))

    return results


if __name__ == "__main__":
    main()
