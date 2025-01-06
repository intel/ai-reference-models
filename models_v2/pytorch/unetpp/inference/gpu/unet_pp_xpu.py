#
# Copyright (c) 2019 Intel Corporation
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
# SPDX-License-Identifier: EPL-2.0
#

import torch
import intel_extension_for_pytorch as ipex

import time
import argparse
import os

from segmentation_models_pytorch.encoders import get_preprocessing_fn
import segmentation_models_pytorch as smp


def parse_option():
    parser = argparse.ArgumentParser("UNet++ dumpy inference")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num-iterations', type=int, default=15, help='Number of iterations to run')
    parser.add_argument('--plain', action='store_true', help='Run using plain format')
    parser.add_argument("--device", type=str, default="xpu")
    parser.add_argument('--dtype', type=str, default="fp16", choices=["fp16"])
    args = parser.parse_args()
    args.profile = os.environ.get("PROFILE", "OFF").upper() in ["1", "Y", "ON", "YES", "TRUE"]
    return args

def create_model(jit=True, dtype=torch.half):
    model = smp.UnetPlusPlus()
    model = model.to(device)
    model.eval()
    model = torch.xpu.optimize(model=model, dtype=dtype)
    if jit:
        with torch.no_grad():
            with torch.autocast("xpu", enabled=True, dtype=dtype):
                print("start tracing")
                model = torch.jit.trace(model, torch.randn(input_size, device="xpu"))
                model = torch.jit.freeze(model)
                print("finish tracing")
    return model

def print_perf_info(total_time, counter, batch_size):
    latency = total_time / counter 
    throughput = batch_size / latency
    print("Latency:", latency)
    print("Throughput:", throughput)

def write_profile_info(prof, bs):
    title = "/unetpp"
    title += "fp16" + "_"
    title += "bs" + str(bs) + "_"
    title = (title + "plain_") if args.plain else (title + "block_")
    # TODO: replace xpu with args.device
    title += "xpu" + "_"

    profiling_path = os.getenv('PROFILE_PATH')
    if not profiling_path:
        profiling_path = ''
    sort_str = "self_" + args.device + "_time_total"
    torch.save(prof.key_averages().table(sort_by=sort_str), profiling_path + title + 'profiling.pt')
    torch.save(prof.table(sort_by="id", row_limit=100000), profiling_path + title + 'profiling_detailed.pt')
    # torch.save(prof.key_averages(group_by_input_shape=True).table(), profiling_path + title + 'profiling_detailed.pt')
    prof.export_chrome_trace(profiling_path + title + 'profiling.json')
    print(prof.key_averages().table(sort_by=sort_str))
    print(prof.key_averages().table(sort_by=sort_str), file=open(profiling_path + title + '.breakdown', 'w'))
    print(prof.key_averages(group_by_input_shape=True).table())


def inference(model, profile=False, num_iteration=100, warm_iter=5, log_iter=5):
    # get dummy data
    rand_input = torch.randn(input_size)
    # randinput
    total_time = 0.0
    counter = 0
    with torch.no_grad():
        for i in range(num_iteration):
            print("iteration: {}/{}".format(i, num_iteration))
            with torch.autograd.profiler_legacy.profile(enabled=profile, use_xpu=True, record_shapes=False) as prof:
                start_time = time.time()
                #h2d
                rand_input = rand_input.to(device)
                output = model(rand_input)

                #d2h
                output = output.cpu()
                torch.xpu.synchronize()
                end_time = time.time()

            batch_time = end_time - start_time
            if i % log_iter == 0:
                print(f"[Test {i} / {num_iteration}, current batch time: {batch_time}]")
            if i >= warm_iter:
                total_time += batch_time
                counter += 1

            if profile:
                write_profile_info(prof, input_size[0])
            if i == num_iteration -1:
                print_perf_info(total_time, counter, input_size[0])
                return

if __name__=="__main__":
    args = parse_option()
    device = "xpu"
    assert args.dtype == "fp16"
    dtype = torch.half
    input_size = (args.batch_size, 3, 512, 512)

    model = create_model(jit=True, dtype=dtype)
    inference(model, profile=args.profile, num_iteration=args.num_iterations)
