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
import os
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./model")
    parser.add_argument("--model_name", type=str, default="int4_weight_pro.pt")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    print(args)

    shard_a = torch.load(f"{args.model_path}/pytorch_model-00001-of-00003.bin")
    shard_b = torch.load(f"{args.model_path}/pytorch_model-00002-of-00003.bin")
    shard_c = torch.load(f"{args.model_path}/pytorch_model-00003-of-00003.bin")
    shard_int4 = torch.load(f"{args.model_path}/int4_weight.pt")

    shard_a.update(shard_b)
    shard_a.update(shard_c)
    shard_a.update(shard_int4)

    shard_b = shard_a

    for i in shard_a:
        if i.find("q_proj.qweight") != -1 or i.find("q_proj.scales") != -1 \
                or i.find("q_proj.qzeros") != -1 or i.find("q_proj.group_size") != -1 \
                or i.find("k_proj.qweight") != -1 or i.find("k_proj.scales") != -1 \
                or i.find("k_proj.qzeros") != -1 or i.find("k_proj.group_size") != -1 \
                or i.find("v_proj.qweight") != -1 or i.find("v_proj.scales") != -1 \
                or i.find("v_proj.qzeros") != -1 or i.find("v_proj.group_size") != -1:
            shard_b[i] = torch.empty([])

    torch.save(shard_b, os.path.join(args.model_path, args.model_name))
    print(f"Model saved to {os.path.join(args.model_path, args.model_name)}")
