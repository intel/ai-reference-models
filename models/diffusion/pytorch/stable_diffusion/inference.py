#
# -*- coding: utf-8 -*-
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
import logging
import os
import time
import numpy as np
import pathlib

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
from torchmetrics.image.fid import FrechetInceptionDistance

logging.getLogger().setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="CompVis/stable-diffusion-v1-4", help="Model path")
    parser.add_argument("--prompt", type=str, default="a photo of an astronaut riding a horse on mars", help="input text")
    parser.add_argument("--output_dir", type=str, default=None,help="output path")
    parser.add_argument("--real_images", type=str, default="pt_results", help="Path to real images for FID input")
    parser.add_argument("--seed", type=int, default=2022, help="random seed")
    parser.add_argument('--precision', type=str, default="fp32", help='precision: fp32, bf16, fp16, int8')
    parser.add_argument('--ipex', action='store_true', default=False, help='ipex')
    parser.add_argument('--trace', action='store_true', default=False, help='jit trace')
    parser.add_argument('--compile_ipex', action='store_true', default=False, help='compile with ipex backend')
    parser.add_argument('--compile_inductor', action='store_true', default=False, help='compile with inductor backend')
    parser.add_argument('--calibration', action='store_true', default=False, help='doing calibration step for int8 path')
    parser.add_argument('--configure-dir', default='configure.json', type=str, metavar='PATH', help = 'path to int8 configures, default file name is configure.json')
    parser.add_argument('--profile', action='store_true', default=False, help='profile')
    parser.add_argument('--benchmark', action='store_true', default=False, help='test performance')
    parser.add_argument('--accuracy', action='store_true', default=False, help='test accuracy')

    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    logging.info(f"Parameters {args}")

    # load model
    pipe = StableDiffusionPipeline.from_pretrained(args.model_name_or_path)

    # data type
    if args.precision == "fp32":
        print("Running fp32 ...")
        dtype=torch.float32
    elif args.precision == "bf16":
        print("Running bf16 ...")
        dtype=torch.bfloat16
    elif args.precision == "fp16":
        print("Running fp16 ...")
        dtype=torch.half
    elif args.precision == "int8":
        print("Running int8 ...")
    else:
        raise ValueError("--precision needs to be the following:: fp32, bf16, fp16, int8")

    input = torch.randn(4, 4, 64, 64), torch.tensor(921), torch.randn(4, 77, 768)

    # ipex
    if args.ipex:
        import intel_extension_for_pytorch as ipex
        if args.precision == "fp32":
            pipe.unet = ipex.optimize(pipe.unet.eval(), inplace=True)
        elif args.precision == "bf16" or args.precision == "fp16":
            pipe.unet = ipex.optimize(pipe.unet.eval(), dtype=dtype, inplace=True)
        elif args.precision == "int8":
                if not args.calibration:
                    qconfig = ipex.quantization.default_static_qconfig
                    pipe.unet = ipex.quantization.prepare(pipe.unet, qconfig, input, inplace=True)
                    pipe.unet.load_qconf_summary(qconf_summary=args.configure_dir)
                    pipe.unet = ipex.quantization.convert(pipe.unet)
                    print("running int8 evalation step\n")
        else:
            raise ValueError("--precision needs to be the following:: fp32, bf16, fp16, int8")
        print("Running IPEX ...")


    # jit trace
    if args.trace:
        # from utils_vis import make_dot, draw
        if args.precision == "bf16" or args.precision == "fp16":
            with torch.cpu.amp.autocast(dtype=dtype), torch.no_grad():
                pipe.unet = torch.jit.trace(pipe.unet, input, strict=False)
                pipe.unet = torch.jit.freeze(pipe.unet)
                pipe.unet(*input)
                pipe.unet(*input)
                # graph = pipe.unet.graph_for(torch.randn(4, 4, 64, 64), torch.tensor(921), torch.randn(4, 77, 768))
                # print(graph)
                # draw(graph).render("stable_diffusion")

        else:
            with torch.no_grad():
                pipe.unet = torch.jit.trace(pipe.unet, input, strict=False)
                pipe.unet = torch.jit.freeze(pipe.unet)
                pipe.unet(*input)
                pipe.unet(*input)
                # graph = pipe.unet.graph_for(torch.randn(4, 4, 64, 64), torch.tensor(921), torch.randn(4, 77, 768))
                # print(graph)
                # draw(graph).render("stable_diffusion")

    # torch compile with ipex backend
    if args.compile_ipex:
        pipe.unet = torch.compile(pipe.unet, backend='ipex')
    # torch compile with inductor backend
    if args.compile_inductor:
        pipe.unet = torch.compile(pipe.unet, backend='inductor')

    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    if args.ipex and args.precision == "int8" and args.calibration:
        print("runing int8 calibration step\n")
        qconfig = ipex.quantization.default_static_qconfig
        pipe.unet = ipex.quantization.prepare(pipe.unet, qconfig, input, inplace=True)
        pipe(args.prompt, generator=generator).images
        pipe.unet.save_qconf_summary(args.configure_dir)
        print(".........calibration step done..........")

    # benchmark
    if args.benchmark:
        # run model
        start = time.time()
        if args.precision == "bf16" or args.precision == "fp16":
            with torch.cpu.amp.autocast(dtype=dtype), torch.no_grad():
                image = pipe(args.prompt, generator=generator).images
        else:
            with torch.no_grad():
                image = pipe(args.prompt, generator=generator).images
        end = time.time()
        print('time per prompt(s): {:.2f}'.format((end - start)))
        if args.output_dir:
            if not os.path.exists(args.output_dir):
                os.mkdir(args.output_dir)
            image_name = time.strftime("%Y%m%d_%H%M%S")
            image[0].save(f"{args.output_dir}/{image_name}.png")
        
    if args.accuracy:
        # run model
        images = []
        for prompt in ["a photo of an astronaut riding a horse on mars", "A high tech solarpunk utopia in the Amazon rainforest"]:
            if args.precision == "bf16" or args.precision == "fp16":
                with torch.cpu.amp.autocast(dtype=dtype), torch.no_grad():
                    image = pipe(prompt, generator=generator, output_type="numpy").images
            else:
                with torch.no_grad():
                    image = pipe(prompt, generator=generator, output_type="numpy").images
            images.append(image[0])

            if args.output_dir:
                if not os.path.exists(args.output_dir):
                    os.mkdir(args.output_dir)
                image_name = time.strftime("%Y%m%d_%H%M%S")
                Image.fromarray((image[0] * 255).round().astype("uint8")).save(f"{args.output_dir}/{image_name}.png")

        fake_images = torch.tensor(images)
        fake_images = fake_images.permute(0, 3, 1, 2)
        
        # load real images
        dataset_path = args.real_images
        image_paths = sorted([os.path.join(dataset_path, x) for x in os.listdir(dataset_path)])
        real_images = [np.array(Image.open(path).convert("RGB")) for path in image_paths]
        
        def preprocess_image(image):
            image = torch.tensor(image).unsqueeze(0)
            return image.permute(0, 3, 1, 2) / 255.0
        real_images = torch.cat([preprocess_image(image) for image in real_images])

        # compute FID
        fid = FrechetInceptionDistance(normalize=True)
        fid.update(real_images, real=True)
        fid.update(fake_images, real=False)

        print(f"FID: {float(fid.compute())}")


    # profile
    if args.profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as p:
            if args.precision == "bf16" or args.precision == "fp16":
                with torch.cpu.amp.autocast(dtype=dtype), torch.no_grad():
                    images = pipe(args.prompt, generator=generator).images
            else:
                with torch.no_grad():
                    images = pipe(args.prompt, generator=generator).images

        output = p.key_averages().table(sort_by="self_cpu_time_total")
        print(output)
        timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
        if not os.path.exists(timeline_dir):
            try:
                os.makedirs(timeline_dir)
            except:
                pass
        timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + '-' + \
                    'stable_diffusion-' + '-' + str(os.getpid()) + '.json'
        p.export_chrome_trace(timeline_file)

if __name__ == '__main__':
    main()
