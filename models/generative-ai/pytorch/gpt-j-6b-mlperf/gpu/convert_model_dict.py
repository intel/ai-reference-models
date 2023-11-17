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
# ============================================================================

import argparse
import json
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./model/int4_weight_pro.pt")
    parser.add_argument("--config_input_path", type=str, default="./configs/pytorch_model.bin.index.json")
    parser.add_argument("--config_output_path", type=str, default="./configs/pytorch_model.bin.index_int4.json")
    args = parser.parse_args()
    print(args)
    return args

if __name__ == "__main__":
    args = parse_args()
    with open(args.config_input_path, 'r') as file:
        src_cfg = json.load(file)

    dst_cfg = {
        "metadata": src_cfg["metadata"],
        "weight_map": {}
    }

    model = torch.load(args.model_path)
    for key in model.keys():
        dst_cfg["weight_map"][key] = "int4_weight_pro.pt"

    dst_str = json.dumps(dst_cfg, indent=2)
    with open(args.config_output_path, 'w') as file:
        file.write(dst_str)
