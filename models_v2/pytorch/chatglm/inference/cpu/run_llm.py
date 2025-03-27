#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
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
import os
import psutil
import argparse
import time
import json
from pathlib import Path
import pathlib
import numpy as np
from itertools import chain
from datasets import load_dataset, load_from_disk
import re
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
)

import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser("LLM generation script", add_help=False)
parser.add_argument(
    "-m",
    "--model-name-or-path",
    default=None,
    type=str,
    required=True,
    help="path to model  or model name in HF hub",
)
parser.add_argument(
    "--revision",
    default=None,
    type=str,
    help="Specify a specific commit version of the model",
)
parser.add_argument(
    "--device",
    type=str,
    choices=["cpu"],
    help="cpu",
    default="cpu",
)
parser.add_argument(
    "--dtype",
    type=str,
    choices=["fp32", "bf16", "fp16", "int8", "bf32"],
    help="bf16 or fp32 or fp16 or int8 or bf32",
    default="fp32",
)
parser.add_argument(
    "--max-new-tokens", default=32, type=int, help="output max new tokens"
)
parser.add_argument("--dataset", nargs="?", default="lambada", const="lambada")
parser.add_argument("--split", nargs="?", default="validation", const="validation")
parser.add_argument("--output_dir", nargs="?", default="./saved_results")
parser.add_argument("--ipex_static_quantize", action="store_true")
parser.add_argument("--ipex", action="store_true")
parser.add_argument("--inductor", action="store_true")
parser.add_argument("--ipex_smooth_quant", "--smooth_quant", action="store_true")
parser.add_argument("--jit", action="store_true")
parser.add_argument(
    "--int8_bf16_mixed",
    action="store_true",
    help="by default it is int8-fp32 mixed, to enable int8 mixed amp bf16 (work on platforms like SPR)",
)
parser.add_argument("--quantized_model_path", default="./best_model.pt")
parser.add_argument("--do-calibration", action="store_true")
parser.add_argument("--int8-qconfig", nargs="?", default="./qconfig.json")
parser.add_argument("--lambada", action="store_true")
parser.add_argument("--accuracy_only", action="store_true")
parser.add_argument("--benchmark", action="store_true")
parser.add_argument("--input-tokens", default="32", type=str)
parser.add_argument("--prompt", default=None, type=str)
parser.add_argument("--num-iter", default=100, type=int, help="num iter")
parser.add_argument("--num-warmup", default=10, type=int, help="num warmup")
parser.add_argument("--batch-size", default=1, type=int, help="batch size")
parser.add_argument("--print-memory", action="store_true")
parser.add_argument("--profile", action="store_true")
parser.add_argument("--token-latency", action="store_true")
parser.add_argument("--use-share-weight", action="store_true")
parser.add_argument(
    "--ws-total-cores", default=56, type=int, help="weight sharing total cores"
)
parser.add_argument(
    "--ws-cores-per-instance",
    default=4,
    type=int,
    help="weight sharing core per instance",
)
args = parser.parse_args()

if args.ipex:
    import intel_extension_for_pytorch as ipex
    from intel_extension_for_pytorch.quantization import convert, prepare


def trace_handler(prof):
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))
    import datetime
    import threading

    now = datetime.datetime.now()
    record_id = threading.current_thread().ident
    log_path = os.path.join(
        args.output_dir,
        "llama_profiling_{}_{}_step_{}.json".format(
            now.strftime("%Y%m%d%H%M%S"), record_id, str(prof.step_num)
        ),
    )
    prof.export_chrome_trace(log_path)


# beam search = 4
num_beams = 4
generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=num_beams)


def get_memory_usage(name, args):
    if args.print_memory:
        memory_allocated = round(psutil.Process().memory_info().rss / 1024**3, 3)
        print(name, "memory used total:", memory_allocated, "GB")
    else:
        return


device = torch.device(args.device)
args.dtype = "int8" if args.int8_bf16_mixed else args.dtype

# amp autocast
if args.dtype == "bf16":
    amp_enabled = True
    amp_dtype = torch.bfloat16
elif args.dtype == "fp16":
    amp_enabled = True
    amp_dtype = torch.half
else:
    amp_enabled = True if args.int8_bf16_mixed else False
    amp_dtype = torch.bfloat16 if args.int8_bf16_mixed else torch.float


has_position_id = False
if "llama" in args.model_name_or_path:
    user_model = LlamaForCausalLM.from_pretrained(
        args.model_name_or_path,
        low_cpu_mem_usage=True,
        torchscript=args.jit,
        revision=args.revision,
    )
    tokenizer = LlamaTokenizer.from_pretrained(
        args.model_name_or_path,
        revision=args.revision,
    )
    if hasattr(user_model.config, "num_key_value_heads"):
        del user_model.config.num_key_value_heads
else:
    user_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        low_cpu_mem_usage=True,
        torchscript=args.jit,
        trust_remote_code=True,
        revision=args.revision,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        revision=args.revision,
    )
get_memory_usage("Host", args)

if args.dtype == "bf16" or args.dtype == "fp32":
    if args.ipex:
        user_model = ipex.llm.optimize(
            user_model.eval(),
            dtype=torch.bfloat16 if args.dtype == "bf16" else torch.float,
            inplace=True,
            deployment_mode=True if args.jit and (not args.accuracy_only) else False,
        )
    if args.inductor:
        from torch._inductor import config as inductor_config

        inductor_config.cpp_wrapper = True
        with (
            torch.no_grad(),
            torch.autocast("cpu", enabled=amp_enabled, dtype=amp_dtype),
        ):
            if args.ipex:
                print("[Info] Running torch.compile() with IPEX backend")
                user_model.forward = torch.compile(
                    user_model.forward, dynamic=True, backend="ipex"
                )
            else:
                print("[Info] Running torch.compile() BFloat16 with default backend")
                user_model.forward = torch.compile(user_model.forward, dynamic=True)
elif args.dtype == "fp16":
    if args.ipex:
        user_model = ipex.llm.optimize(
            user_model.eval(),
            dtype=torch.half,
            inplace=True,
            deployment_mode=True if args.jit and (not args.accuracy_only) else False,
        )
    if args.inductor:
        from torch._inductor import config as inductor_config

        inductor_config.cpp_wrapper = True
        with (
            torch.no_grad(),
            torch.autocast("cpu", enabled=amp_enabled, dtype=amp_dtype),
        ):
            if args.ipex:
                print("[Info] Running torch.compile() with IPEX backend")
                user_model.forward = torch.compile(
                    user_model.forward, dynamic=True, backend="ipex"
                )
            else:
                print("[Info] Running torch.compile() BFloat16 with default backend")
                user_model.forward = torch.compile(user_model.forward, dynamic=True)
elif args.dtype == "bf32":
    ipex.set_fp32_math_mode(mode=ipex.FP32MathMode.BF32, device="cpu")
    user_model = ipex.llm.optimize(
        user_model.eval(),
        dtype=torch.float,
        inplace=True,
        deployment_mode=True if args.jit and (not args.accuracy_only) else False,
    )

get_memory_usage("IPEX", args)
print("Data type of the model:", user_model.dtype)

global_past_key_value = None
if re.search("GPTJ", user_model.config.architectures[0], re.IGNORECASE):
    has_position_id = True
    beam_idx_tmp = torch.zeros(
        (2048, int(args.batch_size * num_beams)), dtype=torch.long
    ).contiguous()
    global_past_key_value = [
        (
            torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
            torch.zeros(
                [
                    1,
                    user_model.config.num_attention_heads,
                    1,
                    int(
                        user_model.config.hidden_size
                        / user_model.config.num_attention_heads
                    ),
                ]
            ).contiguous(),
            torch.zeros(
                [
                    1,
                    user_model.config.num_attention_heads,
                    1,
                    int(
                        user_model.config.hidden_size
                        / user_model.config.num_attention_heads
                    ),
                ]
            ).contiguous(),
            beam_idx_tmp,
        )
        for i in range(user_model.config.num_hidden_layers)
    ]

elif re.search("llama", user_model.config.architectures[0], re.IGNORECASE):
    has_position_id = True
    beam_idx_tmp = torch.zeros(
        (2048, int(args.batch_size * num_beams)), dtype=torch.long
    ).contiguous()
    global_past_key_value = [
        (
            torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
            torch.zeros(
                [
                    1,
                    user_model.config.num_attention_heads,
                    1,
                    int(
                        user_model.config.hidden_size
                        / user_model.config.num_attention_heads
                    ),
                ]
            ).contiguous(),
            torch.zeros(
                [
                    1,
                    user_model.config.num_attention_heads,
                    1,
                    int(
                        user_model.config.hidden_size
                        / user_model.config.num_attention_heads
                    ),
                ]
            ).contiguous(),
            beam_idx_tmp,
        )
        for i in range(user_model.config.num_hidden_layers)
    ]
elif re.search("chatglm", user_model.config.architectures[0], re.IGNORECASE):
    has_position_id = True
    beam_idx_tmp = torch.zeros(
        (2048, int(args.batch_size * num_beams)), dtype=torch.long
    ).contiguous()
    global_past_key_value = [
        (
            torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
            torch.zeros(
                [
                    1,
                    user_model.config.num_attention_heads,
                    1,
                    int(
                        user_model.config.hidden_size
                        / user_model.config.num_attention_heads
                    ),
                ]
            ).contiguous(),
            torch.zeros(
                [
                    1,
                    user_model.config.num_attention_heads,
                    1,
                    int(
                        user_model.config.hidden_size
                        / user_model.config.num_attention_heads
                    ),
                ]
            ).contiguous(),
            beam_idx_tmp,
        )
        for i in range(user_model.config.num_layers)
    ]


if global_past_key_value == None:
    print("This scirpt only supports llama gptj chatglm.")
    exit(0)


class Evaluator:
    def __init__(self, dataset, tokenizer, batch_size=8, pad_val=1, pad_max=196):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.pad_val = pad_val
        self.pad_max = pad_max

        # tokenize the dataset
        self.dataset = self.dataset.map(self.tokenize_function, batched=True)
        self.dataset.set_format(type="torch", columns=["input_ids"])

    @torch.no_grad()
    def tokenize_function(self, examples):
        example = self.tokenizer(examples["text"])
        return example

    @torch.no_grad()
    def collate_batch(self, batch):
        position_ids_padded = []
        input_ids_padded = []
        last_ind = []
        attention_mask_padded = []
        for text in batch:
            # we cut the sentence if it exceeds pad_max, we are using tuned max 196 from gptj model; TODO: tune best pad_max
            input_ids = (
                text["input_ids"]
                if text["input_ids"].shape[0] <= self.pad_max
                else text["input_ids"][0 : int(self.pad_max - 1)]
            )
            pad_len = self.pad_max - input_ids.shape[0]
            last_ind.append(input_ids.shape[0] - 1)
            attention_mask = torch.ones(len(input_ids))
            position_ids = torch.arange(len(input_ids))
            input_ids = pad(input_ids, (0, pad_len), value=self.pad_val)
            input_ids_padded.append(input_ids)
            attention_mask = pad(attention_mask, (0, pad_len), value=0)
            attention_mask_padded.append(attention_mask)
            position_ids = pad(position_ids, (0, pad_len), value=self.pad_val)
            position_ids_padded.append(position_ids)
        if args.inductor:
            return ((torch.vstack(input_ids_padded)), torch.tensor(last_ind))
        else:
            if has_position_id:
                return (
                    (
                        torch.vstack(input_ids_padded),
                        torch.vstack(attention_mask_padded),
                        tuple(global_past_key_value),
                        torch.vstack(position_ids_padded),
                    ),
                    torch.tensor(last_ind),
                )
            else:
                return (
                    (
                        torch.vstack(input_ids_padded),
                        torch.vstack(attention_mask_padded),
                        tuple(global_past_key_value),
                        None,
                    ),
                    torch.tensor(last_ind),
                )

    @torch.no_grad()
    def evaluate(self, model):
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        latency = 0
        test_dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_batch,
        )

        if args.inductor:
            for i, ((input_ids), last_ind) in enumerate(test_dataloader):
                label = input_ids[torch.arange(len(last_ind)), last_ind]
                input_ids[torch.arange(len(last_ind)), last_ind] = self.pad_val
                pad_len = self.pad_max - last_ind - 1
                start = time.time()
                outputs = model(
                    input_ids,
                )
                latency += time.time() - start
                if isinstance(outputs, tuple):
                    res = outputs[0]
                else:
                    res = outputs["logits"]
                last_token_logits = res[torch.arange(len(last_ind)), -2 - pad_len, :]

                pred = last_token_logits.argmax(dim=-1)
                total += label.size(0)
                hit += (pred == label).sum().item()
                if i % 50 == 0:
                    print(hit / total)
                    print("Processed minibatch:", i)

            acc = hit / total
            print(acc)
            lantecy = latency / len(self.dataset)
            return acc, lantecy
        else:
            for i, (
                (input_ids, attention_mask, past_key_values, position_ids),
                last_ind,
            ) in enumerate(test_dataloader):
                label = input_ids[torch.arange(len(last_ind)), last_ind]
                input_ids[torch.arange(len(last_ind)), last_ind] = self.pad_val
                pad_len = self.pad_max - last_ind - 1
                start = time.time()
                if has_position_id:
                    outputs = model(
                        input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        return_last_logit=torch.tensor(True),
                    )
                else:
                    outputs = model(
                        input_ids,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        return_last_logit=torch.tensor(True),
                    )

                latency += time.time() - start

                last_token_logits = outputs[0][
                    torch.arange(len(last_ind)), -2 - pad_len, :
                ]

                pred = last_token_logits.argmax(dim=-1)
                total += label.size(0)
                hit += (pred == label).sum().item()
                if i % 50 == 0:
                    print(hit / total)
                    print("Processed minibatch:", i)

            acc = hit / total
            print(acc)
            lantecy = latency / len(self.dataset)
            return acc, lantecy


if args.lambada:
    full_dataset = load_dataset(args.dataset)
    dataset = full_dataset["validation"]
    calib_dataset = full_dataset["train"]

    user_model.eval()
    evaluator = Evaluator(dataset, tokenizer, 1)
    calib_evaluator = Evaluator(calib_dataset, tokenizer, 1)

    calib_dataloader = DataLoader(
        calib_evaluator.dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=evaluator.collate_batch,
    )

    test_dataloader = DataLoader(
        evaluator.dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=evaluator.collate_batch,
    )


def calib_func(prepared_model):
    for i, (
        (input_ids, attention_mask, past_key_values, position_ids),
        last_ind,
    ) in enumerate(calib_dataloader):
        if i == 8:
            break
        if has_position_id:
            prepared_model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
            )
        else:
            prepared_model(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )


def eval_func(traced_model):
    print("Evaluating LLM")
    acc, latency = evaluator.evaluate(traced_model)
    print("Accuracy:", acc)
    print("Latency (sec):", latency)
    return acc


# input prompt
current_path = pathlib.Path(__file__).parent.resolve()

if args.prompt is not None:
    prompt = args.prompt
else:
    with open(str(current_path) + "/prompt.json") as f:
        prompt_pool = json.load(f)
    if "chatglm" in prompt_pool and args.input_tokens in prompt_pool["chatglm"]:
        prompt = prompt_pool["chatglm"][args.input_tokens]
    else:
        raise SystemExit(
            "[ERROR] No such input_tokens prompt in prompt.json, Plese use --prompt if want to use custom input."
        )

if args.benchmark:
    input_size = tokenizer(prompt, return_tensors="pt").input_ids.size(dim=1)
    print("---- Prompt size:", input_size)

    if args.token_latency:
        if not hasattr(user_model.config, "token_latency"):
            user_model.config.token_latency = True
    prompt = [prompt] * args.batch_size

if args.dtype == "int8" and args.ipex:
    qconfig = ipex.quantization.default_static_qconfig_mapping
    if args.ipex_smooth_quant:
        qconfig = ipex.quantization.get_smooth_quant_qconfig_mapping(alpha=0.75)

    if args.do_calibration:
        example_inputs = None
        for i, (
            (input_ids, attention_mask, past_key_values, position_ids),
            last_ind,
        ) in enumerate(calib_dataloader):
            example_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
            }
            if has_position_id:
                example_inputs["position_ids"] = position_ids
            break
        from intel_extension_for_pytorch.quantization import prepare, convert

        user_model = ipex.llm.optimize(
            user_model.eval(),
            dtype=amp_dtype,
            quantization_config=qconfig,
            inplace=True,
            deployment_mode=False,
        )
        prepared_model = prepare(
            user_model.eval(), qconfig, example_kwarg_inputs=example_inputs
        )
        with torch.no_grad():
            for i, (
                (input_ids, attention_mask, past_key_values, position_ids),
                last_ind,
            ) in enumerate(calib_dataloader):
                if i == 8:
                    break
                if has_position_id:
                    prepared_model(
                        input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                    )
                else:
                    prepared_model(
                        input_ids,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                    )
        prepared_model.save_qconf_summary(qconf_summary=args.int8_qconfig)
        print("calibration Done!")
    else:
        user_model = ipex.llm.optimize(
            user_model.eval(),
            dtype=amp_dtype,
            quantization_config=qconfig,
            qconfig_summary_file=args.int8_qconfig,
            inplace=True,
            deployment_mode=True,
        )
        print("model quantization - Done!")
elif args.dtype == "int8" and args.inductor:
    from torch._inductor import config as inductor_config

    inductor_config.cpp_wrapper = True

    if args.ipex_smooth_quant:
        from torchao.prototype.smoothquant import (
            insert_smooth_quant_observer_,
            SmoothQuantObservedLinear,
            smooth_quant,
        )
        from torchao.quantization import quantize_

        with torch.no_grad():
            encoded_input = tokenizer(prompt, return_tensors="pt")
            print("encoded_input is: {}".format(encoded_input), flush=True)
            insert_smooth_quant_observer_(user_model, alpha=0.5, quant_mode="static")
            for i in range(3):
                user_model(**encoded_input)
            is_observed_linear = lambda m, fqn: isinstance(m, SmoothQuantObservedLinear)
            print(f"running SmoothQuant with static quantization")
            quantize_(user_model, smooth_quant(), is_observed_linear)
            with torch.autocast("cpu", enabled=amp_enabled, dtype=amp_dtype):
                user_model = torch.compile(user_model, dynamic=True)
                user_model(**encoded_input)
                user_model(**encoded_input)
    else:
        from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
        import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
        from torch.ao.quantization.quantizer.x86_inductor_quantizer import (
            X86InductorQuantizer,
        )
        from torch._export import export_for_training

        print("[Info] Running torch.compile() INT8 quantization")
        with torch.no_grad():
            encoded_input = tokenizer(prompt, return_tensors="pt")
            print("encoded_input is: {}".format(encoded_input), flush=True)
            exported_model = export_for_training(
                user_model,
                (),
                kwargs=encoded_input.data,
            ).module()
            quantizer = X86InductorQuantizer()
            quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())
            prepared_model = prepare_pt2e(exported_model, quantizer)
            prepared_model(**encoded_input)
            converted_model = convert_pt2e(prepared_model)
            torch.ao.quantization.move_exported_model_to_eval(converted_model)
            with torch.autocast("cpu", enabled=amp_enabled, dtype=amp_dtype):
                if args.ipex:
                    print("[Info] Running torch.compile() with IPEX backend")
                    user_model = torch.compile(
                        converted_model, dynamic=True, backend="ipex"
                    )
                else:
                    print("[Info] Running torch.compile() with default backend")
                    user_model = torch.compile(converted_model, dynamic=True)
                user_model(**encoded_input)
                user_model(**encoded_input)


def benchmark_warmup(prompt):
    # start
    if args.profile:

        with (
            torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                schedule=torch.profiler.schedule(wait=1, warmup=3, active=1),
                on_trace_ready=trace_handler,
            ) as prof,
            torch.inference_mode(),
            torch.no_grad(),
            torch.autocast("cpu", enabled=amp_enabled, dtype=amp_dtype),
        ):
            for i in range(5):
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids
                output = user_model.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=args.max_new_tokens,
                    **generate_kwargs,
                )
                prof.step()

    num_iter = args.num_warmup
    with (
        torch.inference_mode(),
        torch.no_grad(),
        torch.autocast("cpu", enabled=amp_enabled, dtype=amp_dtype),
    ):
        for i in range(num_iter):
            get_memory_usage("Iteration: " + str(i), args)

            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            output = user_model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.max_new_tokens,
                **generate_kwargs,
            )
    print("warmup done")


def benchmark_evaluate(prompt):
    # start
    total_time = 0.0
    num_iter = args.num_iter - args.num_warmup
    total_list = []
    with (
        torch.inference_mode(),
        torch.no_grad(),
        torch.autocast("cpu", enabled=amp_enabled, dtype=amp_dtype),
    ):
        for i in range(num_iter):
            get_memory_usage("Iteration: " + str(i), args)
            tic = time.time()
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
            output = user_model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.max_new_tokens,
                **generate_kwargs,
            )
            gen_ids = output[0] if args.token_latency else output
            gen_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            toc = time.time()
            print(gen_text, flush=True)
            total_time += toc - tic
            if args.token_latency:
                total_list.append(output[1])

    print("\n", "-" * 10, "Summary:", "-" * 10)
    latency = total_time / (num_iter)
    print("inference-latency: %.3f sec." % latency)
    if args.token_latency:
        first_latency = np.mean([x[0] for x in total_list])
        next_latency_list = list(chain(*[x[1:] for x in total_list]))
        next_latency_list.sort()
        average_next_latency = np.mean(next_latency_list)
        p90_latency = np.percentile(next_latency_list, 90)
        print("first-token-latency: %.3f sec." % first_latency)
        print("rest-token-latency: %.3f sec." % average_next_latency)
        print("P90-rest-token-latency: %.3f sec." % p90_latency)


if args.benchmark:
    torch._C._jit_set_texpr_fuser_enabled(False)
    benchmark_warmup(prompt)
    if args.use_share_weight and args.device == "cpu":
        threads = []
        import threading

        num_instances = args.ws_total_cores // args.ws_cores_per_instance
        for i in range(0, num_instances):
            t = threading.Thread(target=benchmark_evaluate, args=(prompt))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
    else:
        benchmark_evaluate(prompt)

if args.accuracy_only:
    if not args.inductor:
        if args.dtype == "int8" or args.int8_bf16_mixed:
            user_model = user_model.trace_graph

        if args.jit and (
            args.dtype == "bf16"
            or args.dtype == "fp32"
            or args.dtype == "bf32"
            or args.dtype == "fp16"
        ):
            input_ids = torch.ones(32).to(torch.long)
            attention_mask = torch.ones(len(input_ids))
            position_ids = torch.arange(len(input_ids))
            last_ind = input_ids.shape[0] - 1
            example_inputs = (
                {
                    "input_ids": input_ids.unsqueeze(0),
                    "attention_mask": attention_mask.unsqueeze(0),
                    "position_ids": position_ids.unsqueeze(0),
                    "past_key_values": tuple(global_past_key_value),
                    "return_last_logit": torch.tensor(True),
                }
                if has_position_id
                else {
                    "input_ids": input_ids.unsqueeze(0),
                    "attention_mask": attention_mask.unsqueeze(0),
                    "past_key_values": tuple(global_past_key_value),
                    "return_last_logit": torch.tensor(True),
                }
            )
            with (
                torch.no_grad(),
                torch.autocast("cpu", enabled=amp_enabled, dtype=amp_dtype),
            ):
                user_model = torch.jit.trace(
                    user_model.eval(),
                    example_kwarg_inputs=example_inputs,
                    strict=False,
                    check_trace=False,
                )
                user_model = torch.jit.freeze(user_model.eval())

    with torch.autocast("cpu", enabled=amp_enabled, dtype=amp_dtype):
        eval_func(user_model)
