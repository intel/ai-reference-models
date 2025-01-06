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

import contextlib
import os
import time
import sys
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import argparse
import numpy as np
from scipy.linalg import sqrtm
from PIL import Image
import pytorch_fid
import requests

from datasets import load_dataset

from torchmetrics.functional.multimodal import clip_score
from functools import partial

parser = argparse.ArgumentParser(description='PyTorch StableDiffusion TexttoImage')
parser.add_argument('--prompt', default="nateraw/parti-prompts", type=str, help='prompt_dataset')
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--idx_start', default=0, type=int, help='select the start index of image')
parser.add_argument('--precision', choices=["fp32", "fp16", "bf16"],
                    default="fp16", type=str, help='precision')
parser.add_argument('--jit', action='store_true', default=False, help='enable JIT')
parser.add_argument('--iteration', default=30, type=int, help='test iterations')
parser.add_argument('--warmup_iter', default=2, type=int, help='test warmup')
parser.add_argument('--device', default='xpu', type=str, help='cpu, cuda or xpu')
parser.add_argument('--save_image', action='store_true', default=False, help='save image')
parser.add_argument('--save_tensor', action='store_true', default=False, help='save tensor')
parser.add_argument('--accuracy', action='store_true', default=False, help='compare the result with cuda')
parser.add_argument('-m', '--model_id',
                    choices=["CompVis/stable-diffusion-v1-4", "stabilityai/stable-diffusion-2-1"],
                    default='stabilityai/stable-diffusion-2-1', type=str, metavar='PATH',
                    help='path to model structure or weight')
parser.add_argument('--ref_path', default='', type=str, metavar='PATH',
                    help='path to reference image (default: none)')
parser.add_argument('--save_path', default='./xpu_result', type=str, help='output image dir')
parser.add_argument('--num_inference_steps', default=50, type=int, help='number of unet step')
parser.add_argument("--disable_optimize_transformers", action="store_true")
parser.add_argument('--evaluate_method', choices=["clip", "fid"],
                    default="fid", type=str, help='evaluation method, now we suppor clip and fid')
parser.add_argument('--pipeline_mode', choices=["img2img", "text2img"],
                    default="text2img", type=str, help='evaluation method, now we suppor clip and fid')
parser.add_argument('--channels_last', action='store_true', default=False, help='enable channels_last')

args = parser.parse_args()
print(args)

def compare(xpu_res, ref_res):
    xpu = torch.tensor(xpu_res)
    ref = torch.tensor(ref_res)

    diff_value = torch.abs((xpu - ref))
    max_diff = torch.max(diff_value)

    shape = 1
    for i in range(xpu.dim()):
        shape = shape * xpu.shape[i]

    value = diff_value > 0.1
    num = torch.sum(value.contiguous().view(-1))
    ratio1 = num / shape
    print("difference larger than 0.1, ratio = {}".format(ratio1)) 

    value = diff_value > 0.01
    num = torch.sum(value.contiguous().view(-1))
    ratio2 = num / shape
    print("difference larger than 0.01, ratio = {}".format(ratio2))

    value = diff_value > 0.001
    num = torch.sum(value.contiguous().view(-1))
    ratio3 = num / shape
    print("difference larger than 0.001, ratio = {}".format(ratio3))

    if ratio1 < 0.01 and ratio2 < 0.08 and ratio3 < 0.4:
        print("accuracy pass")
    else:
        print("accuracy fail")

def compare_pil_images(ref_res, cur_res):
    xpu = torch.tensor(np.array(cur_res))
    ref = torch.tensor(np.array(ref_res))

    diff_value = torch.abs((xpu - ref))
    max_diff = torch.max(diff_value)

    shape = 1
    for i in range(xpu.dim()):
        shape = shape * xpu.shape[i]

    value = diff_value > 0.1
    num = torch.sum(value.contiguous().view(-1))
    ratio1 = num / shape
    print("difference larger than 0.1, ratio = {}".format(ratio1))

    value = diff_value > 0.01
    num = torch.sum(value.contiguous().view(-1))
    ratio2 = num / shape
    print("difference larger than 0.01, ratio = {}".format(ratio2))

    value = diff_value > 0.001
    num = torch.sum(value.contiguous().view(-1))
    ratio3 = num / shape
    print("difference larger than 0.001, ratio = {}".format(ratio3))

    if ratio1 < 0.01 and ratio2 < 0.08 and ratio3 < 0.4:
        print("accuracy pass")
    else:
        print("accuracy fail")

def main():
    def calculate_clip_score(images, prompts):
        images_int = (images * 255).astype("uint8")
        clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
        return round(float(clip_score), 4)

    profiling = os.environ.get("PROFILE", "OFF").upper() in ["1", "Y", "ON", "YES", "TRUE"]

    # prompt = ["A painting of a squirrel eating a burger"]
    prompts_dataset = load_dataset(args.prompt, split="train")

    seed = 666
    prompts_dataset = prompts_dataset.shuffle(seed=seed)
    is_arc = False

    if args.device == "xpu":
        import intel_extension_for_pytorch as ipex
        idx = torch.xpu.current_device()
        is_arc = torch.xpu.get_device_name(idx)=='Intel(R) Arc(TM) Graphics'
        generator = torch.xpu.default_generators[idx]
        generator.manual_seed(seed)
    elif args.device == "cuda":
        generator = torch.Generator(device=args.device).manual_seed(seed)
    else:
        generator = torch.Generator(device=args.device)

    amp_enabled = False
    if args.precision == "fp32":
        datatype = torch.float
    elif args.precision == "fp16":
        datatype = torch.float16
        amp_enabled = True
    elif args.precision == "bf16":
        datatype = torch.bfloat16
        amp_enabled = True
    else:
        print("unsupported datatype")
        sys.exit()

    if args.pipeline_mode == "img2img":
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(args.model_id, torch_dtype=datatype)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=datatype)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(args.device)

    if args.device == "xpu":
        pipe.unet = torch.xpu.optimize(pipe.unet.eval(), dtype=datatype, inplace=True)
        pipe.vae = torch.xpu.optimize(pipe.vae.eval(), dtype=datatype, inplace=True)
        pipe.text_encoder = torch.xpu.optimize(pipe.text_encoder.eval(), dtype=datatype, inplace=True)

        if not args.disable_optimize_transformers and args.precision == "fp16":
            # optimize with ipex
            pipe.unet = ipex.optimize_transformers(pipe.unet.eval(), dtype=datatype, device=args.device, inplace=True)
            print("---- Use ipex optimize_transformers fp16 model.")
        else:
            # optimize with ipex
            print("---- Use ipex optimize model.")

    if args.channels_last or is_arc:
        pipe.unet = pipe.unet.to(memory_format=torch.channels_last)
        pipe.vae = pipe.vae.to(memory_format=torch.channels_last)
        pipe.text_encoder = pipe.text_encoder.to(memory_format=torch.channels_last)

    if args.evaluate_method == "clip":
        clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")
        out_type = "np"
        clip_score_list = []
    else:
        out_type = "pil"

    if args.accuracy or args.save_tensor:
        out_type = "tensor"

    if args.pipeline_mode == "img2img":
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        init_image = Image.open(requests.get(url, stream=True).raw)
        prompt = "two tigers"
    total_time = 0
    print("output type is: ", out_type)
    with torch.no_grad():
        for step in range(args.warmup_iter):
            idx1 = args.idx_start + int(step * args.batch_size)
            idx2 = args.idx_start + int((step + 1) * args.batch_size)
            input = prompts_dataset[step]["Prompt"]
            print("input is : ", prompt if args.pipeline_mode == "img2img" else input)
            if args.device == "xpu":
                with torch.autocast("xpu", enabled=amp_enabled, dtype=datatype):
                    if args.pipeline_mode == "img2img":
                        images = pipe(prompt=prompt, image=init_image, generator=generator, num_inference_steps=args.num_inference_steps, output_type=out_type).images
                    else:
                        images = pipe(input, generator=generator, num_inference_steps=args.num_inference_steps, output_type=out_type).images
                torch.xpu.synchronize()
            elif args.device == "cuda":
                images = pipe(input, generator=generator, num_inference_steps=args.num_inference_steps, output_type=out_type).images
                torch.cuda.synchronize()
            else:
                images = pipe(input, generator=generator, num_inference_steps=args.num_inference_steps, output_type=out_type).images

        image_before = []
        iter = 0
        for iter in range(args.iteration):
            print("Iteration = {}".format(iter))
            step = 0
            idx1 = args.idx_start + int(step * args.batch_size)
            idx2 = args.idx_start + int((step + 1) * args.batch_size)
            print("idx1={}".format(idx1))
            print("idx2={}".format(idx2))

            input = prompts_dataset[iter]["Prompt"]
            print("input is : ", prompt if args.pipeline_mode == "img2img" else input)

            if args.device == "xpu":
                with (
                    contextlib.nullcontext(None) if not profiling else
                    torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU,
                                    torch.profiler.ProfilerActivity.XPU],
                        record_shapes=True,
                    )
                ) as prof:
                    try:
                        import memory_check
                        memory_check.display_mem("xpu:0")
                    except:
                        pass
                    start_time = time.time()
                    with torch.autocast("xpu", enabled=amp_enabled, dtype=datatype):
                        if args.pipeline_mode == "img2img":
                            images = pipe(prompt=prompt, image=init_image, generator=generator, num_inference_steps=args.num_inference_steps, output_type=out_type).images
                        else:
                            images = pipe(input, generator=generator, num_inference_steps=args.num_inference_steps, output_type=out_type).images
                    torch.xpu.synchronize()
                    end_time = time.time()
                if profiling:
                    torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"), 'stale_diffusion_inf_profile.pt')
                    # Cannot sort by id when using kineto
                    # torch.save(prof.table(sort_by="id", row_limit=-1), 'stable_diffusion_inf_profile_detailed.pt')
                    prof.export_chrome_trace('./stable_diffusion_inf_profile_trace.json')
            elif args.device == "cuda":
                start_time = time.time()
                images = pipe(input, generator=generator, num_inference_steps=args.num_inference_steps, output_type=out_type).images
                torch.cuda.synchronize()
                end_time = time.time()
            else:
                start_time = time.time()
                images = pipe(input, generator=generator, num_inference_steps=args.num_inference_steps, output_type=out_type).images
                end_time = time.time()


            iter_time = end_time - start_time
            total_time += iter_time
            # latency = total_time / (step + 1)
            # throughput = args.batch_size / latency
            # print("---latency={} s".format(latency))
            # print("---throughput={} fps".format(throughput))

            if args.evaluate_method == "clip":
                sd_clip_score = calculate_clip_score(images, input)
                clip_score_list.append(sd_clip_score)
            else:
                if args.accuracy:
                    for i in range(args.batch_size):
                        name = "result_{}_{}.png".format(idx1 + i, iter) if args.save_image else "result_{}_{}.pt".format(idx1 + i, iter)
                        name = os.path.join(args.ref_path, name)
                        if args.save_image:
                            ref_image = Image.open(name)
                            compare_pil_images(ref_image, images[i])
                        else:
                            ref_pt = torch.load(name)
                            compare(ref_pt, images[i])

                if not os.path.exists(args.save_path):
                    os.mkdir(args.save_path)

                if args.save_tensor:
                    for i in range(args.batch_size):
                        file_name = "./result_{}_{}.pt".format(idx1 + i, iter)
                        save_path = os.path.join(args.save_path, file_name)
                        torch.save(images[i], save_path)

                if args.save_image:
                    for i in range(args.batch_size):
                        file_name = "./result_{}_{}.png".format(idx1 + i, iter)
                        save_path = os.path.join(args.save_path, file_name)
                        images[i].save(save_path)
            iter += 1

        total_sample = args.iteration * args.batch_size
        latency = total_time / total_sample * 1000
        throughput = total_sample / total_time
        print("inference Latency: {} ms".format(latency))
        print("inference Throughput: {} samples/s".format(throughput))
        if args.evaluate_method == "clip":
            print(f"CLIP score: {np.mean(clip_score_list)}")


if __name__ == '__main__':
    main()
