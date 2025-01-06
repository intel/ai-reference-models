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
import argparse
import re
import time
import json
import os
import pathlib
import torch
import types
from pathlib import Path
from datasets import load_dataset
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig
import transformers
import intel_extension_for_pytorch as ipex

import numpy as np
from itertools import chain


from dataset import Dataset

calib_size = 1

torch._C._jit_set_texpr_fuser_enabled(False)
def quantize(args):

    user_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        low_cpu_mem_usage=True,
        return_dict=False
    )
    user_model = user_model.to(memory_format=torch.channels_last)
    user_model.eval()
    if args.dtype == "int4":
        user_model.config.weight_only_quantization = True
    user_model = ipex._optimize_transformers(
        user_model.eval(), dtype=torch.int8, inplace=True
    )

    if args.dtype == "int8":

        calib_dataset = Dataset(dataset_path=args.cal_data_path,model_checkpoint_path=args.model,total_sample_count=args.calib_iters, pad_inputs=args.pad_inputs)
        calib_dataset.loadDataset()
        example_batch = calib_dataset[0]
        input_ids, past_key_values, position_ids, attention_mask = calib_dataset.collate_batch([example_batch])[0]
        example_inputs = (input_ids, attention_mask, position_ids, past_key_values)

        calib_dataloader=DataLoader(calib_dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=calib_dataset.collate_batch
        )

        def calib_func(prepared_model):
            for i, (
                (input_ids, past_key_values, position_ids, attention_mask),
                last_ind,
            ) in enumerate(calib_dataloader):
                if i >= args.calib_iters:
                    break
                prepared_model(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )

        from neural_compressor import PostTrainingQuantConfig, quantization

        op_type_dict = {
            "add": {"weight": {"dtype": ["fp32"]}, "activation": {"dtype": ["fp32"]}},
            "linear": {
                "weight": {
                    "dtype": ["int8"],
                    "scheme": ["sym"],
                    "granularity": ["per_channel"],
                    "algorithm": ["minmax"],
                },
                "activation": {
                    "dtype": ["uint8"],
                    "scheme": ["asym"],
                    "granularity": ["per_tensor"],
                    "algorithm": ["kl"],
                },
            },
        }

        excluded_precisions = []
        if args.sq:
            args.alpha = args.alpha if args.alpha == "auto" else float(args.alpha)
            sq_recipes = {"smooth_quant": True, "smooth_quant_args": {"alpha": args.alpha, "folding": True}}
            #sq_recipes = {"smooth_quant": True, "smooth_quant_args": {'alpha': 'auto', 'folding': False }}
            conf = PostTrainingQuantConfig(
                backend="ipex",
                excluded_precisions=excluded_precisions,
                op_type_dict=op_type_dict,
                recipes=sq_recipes,
                example_inputs=example_inputs,
            )
        else:
            conf = PostTrainingQuantConfig(
                backend="ipex",
                excluded_precisions=excluded_precisions,
                op_type_dict=op_type_dict,
                example_inputs=example_inputs,
            )

        # save config
        user_model.config.save_pretrained(args.output_dir)
        q_model = quantization.fit(
            user_model,
            conf,
            calib_dataloader=calib_dataloader,
            calib_func=calib_func,
        )

        os.makedirs(args.output_dir, exist_ok=True)
        q_model.save(args.output_dir)
    else:
        from optimum.utils import NormalizedConfigManager
        user_model = ipex._optimize_transformers(
                    user_model.eval(), dtype=torch.int8, inplace=True
                    )
        torch._C._jit_set_texpr_fuser_enabled(False)

        def generate_dummy_past_key_values(input_bs, user_model):
            normalized_config = NormalizedConfigManager.get_normalized_config_class(
                user_model.config.model_type
            )(user_model.config)
            nb_pkv = 2
            num_layers = normalized_config.num_layers
            num_attention_heads = normalized_config.num_attention_heads
            hidden_size = normalized_config.hidden_size
            d_k = hidden_size // num_attention_heads
            beam_idx_tmp=torch.zeros((2048, int(4)), dtype=torch.long).contiguous()
            if user_model.config.model_type != "bloom":
                new_shape = [1, num_attention_heads, 1, d_k]
                empty_tensor = torch.zeros(new_shape)
                pkv = (
                    torch.zeros([1, 16, 1, 256]).contiguous().bfloat16(),
                    torch.zeros([1, 16, 1, 256]).contiguous().bfloat16(),
                    beam_idx_tmp,
                    torch.zeros(1, dtype=torch.long).contiguous()
                )

            else:
                pkv = ()
                for nb_pkv in range(nb_pkv):
                    if nb_pkv % 2 == 0:
                        new_shape = [input_bs * num_attention_heads, d_k, 0]
                    else:
                        new_shape = [input_bs * num_attention_heads, 0, d_k]
                    pkv = pkv + (torch.empty(size=new_shape),)
            past_key_values = tuple([(pkv) for _ in range(num_layers)])
            return past_key_values


        example_inputs=None
        input_ids = torch.ones(32).to(torch.long)
        attention_mask = torch.ones(len(input_ids))
        last_ind = input_ids.shape[0] - 1
        position_ids = torch.arange(len(input_ids))
        past_key_values = generate_dummy_past_key_values(1, user_model)
        example_inputs=(input_ids.unsqueeze(0), attention_mask.unsqueeze(0), position_ids.unsqueeze(0), tuple(past_key_values))

        def load_int4_weight_and_convert_woq(model, qconfig, state_dict, inplace=True):
            import copy
            from intel_extension_for_pytorch.nn.modules import IpexWoqLinear

            def _convert(mod, attr_name):
                if isinstance(mod, torch.nn.Linear):
                    mod.qconfig = qconfig.global_qconfig
                    # deal with concat linear
                    if attr_name.endswith('concat_qkv'):
                        attr_base = '.'.join(attr_name.split('.')[:-1])
                        w_q = state_dict[attr_base + '.q_proj.qweight']
                        s_q = state_dict[attr_base + '.q_proj.scales']
                        z_q = state_dict[attr_base + '.q_proj.qzeros']
                        w_k = state_dict[attr_base + '.k_proj.qweight']
                        s_k = state_dict[attr_base + '.k_proj.scales']
                        z_k = state_dict[attr_base + '.k_proj.qzeros']
                        w_v = state_dict[attr_base + '.v_proj.qweight']
                        s_v = state_dict[attr_base + '.v_proj.scales']
                        z_v = state_dict[attr_base + '.v_proj.qzeros']
                        w = torch.cat([w_q, w_k, w_v], dim=0)
                        s = torch.cat([s_q, s_k, s_v], dim=-1)
                        z = torch.cat([z_q, z_k, z_v], dim=-1)
                        mod_new = IpexWoqLinear.from_float_and_int4_weight(mod, w, s, z)
                    else:
                        qweight = state_dict[attr_name + '.qweight']
                        scales = state_dict[attr_name + '.scales']
                        qzeros = state_dict[attr_name + '.qzeros']
                        mod_new = IpexWoqLinear.from_float_and_int4_weight(mod, qweight.float(), scales, qzeros)
                    return mod_new
                mod_new = mod

                for name, child in mod.named_children():
                    attr = attr_name + "." + name if attr_name != "" else name
                    setattr(mod_new, name, _convert(child, attr))
                return mod_new

            if not inplace:
                model_new = copy.deepcopy(model)
            else:
                model_new = model
            return _convert(model_new, "")

        lowp_mode = ipex.quantization.WoqLowpMode.INT8
        qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
            lowp_mode=lowp_mode
        )
        state_dict = torch.load(args.state_dict_file)
        os.makedirs(args.output_dir, exist_ok=True)
        convert_model = load_int4_weight_and_convert_woq(user_model.eval(), qconfig, state_dict)
        with torch.no_grad(), torch.autocast("cpu", enabled=True, dtype=torch.bfloat16):
            convert_model = ipex.optimize(convert_model, dtype=torch.bfloat16, inplace=True, concat_linear=False)
            self_jit = torch.jit.trace(convert_model.eval(), example_inputs, strict=False)
            self_jit = torch.jit.freeze(self_jit.eval())
            self_jit.save(args.output_dir + "/best_int4_model.pt")

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
		"--model", nargs="?", default="EleutherAI/gpt-j-6B", const="EleutherAI/gpt-j-6B"
	)
    parser.add_argument(
		"--dataset", nargs="?", default="NeelNanda/pile-10k", const="NeelNanda/pile-10k"
	)
    parser.add_argument("--dtype", type=str, default="int8")
    parser.add_argument(
		"--max-new-tokens", default=32, type=int, help="output max new tokens"
	)
    parser.add_argument("--output_dir", nargs="?", default="./saved_results")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--ipex", action="store_true")
    parser.add_argument("--sq", action="store_true")
    parser.add_argument("--alpha", default="auto", help="Smooth quant parameter.")
    parser.add_argument(
		"--pad_max_length", default=512, type=int, help="Pad input ids to max length."
	)
    parser.add_argument("--calib_iters", default=512, type=int, help="calibration iters.")
    parser.add_argument("--int8", action="store_true")
    parser.add_argument(
		"--int8_bf16_mixed",
		action="store_true",
		help="by default it is int8-fp32 mixed, to enable int8 mixed amp bf16 (work on platforms like EMR)",
	)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--iters", default=100, type=int, help="num iter")
    parser.add_argument("--num_warmup", default=3, type=int, help="num warmup")
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument("--cal-data-path", help="Path to calibration json file")
    parser.add_argument("--val-data-path", help="Path to validation json file")
    parser.add_argument("--beams", type=int, help="Number of beams for decoder", default=4)
    parser.add_argument("--warmup", action="store_true", help="Do warmup")
    parser.add_argument("--pad-inputs", action="store_true", help="Whether to pad input sequence")
    parser.add_argument("--state-dict-file", type=str, default='', help='State dict file generated by GPTQ.')

    args = parser.parse_args()

    quantize(args)


if __name__=="__main__":

    main()
