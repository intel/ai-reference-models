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
import copy
import logging
import os
import time
import threading
import numpy as np
import pathlib
from tqdm import tqdm

import torch
from PIL import Image
from diffusers import DiffusionPipeline
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.datasets as dset
import torchvision.transforms as transforms

logging.getLogger().setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="stabilityai/stable-diffusion-2-1", help="Model path")
    parser.add_argument("--int8_model_path", type=str, default="sd_v2_1.pt", help="INT8 model path")
    parser.add_argument("--dataset_path", type=str, default=None, help="COCO2017 dataset path")
    parser.add_argument("--prompt", type=str, default="A big burly grizzly bear is show with grass in the background.", help="input text")
    parser.add_argument("--output_dir", type=str, default=None,help="output path")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument('--precision', type=str, default="fp32", help='precision: fp32, bf32, bf16, fp16, int8-bf16, int8-fp32')
    parser.add_argument('--calibration', action='store_true', default=False, help='doing calibration step for LCM int8')
    parser.add_argument('--configure-dir', default='configure.json', type=str, metavar='PATH', help = 'path to LCM int8 configures, default file name is configure.json')
    parser.add_argument('--ipex', action='store_true', default=False, help='ipex')
    parser.add_argument('--jit', action='store_true', default=False, help='jit trace')
    parser.add_argument('--compile_ipex', action='store_true', default=False, help='compile with ipex backend')
    parser.add_argument('--compile_inductor', action='store_true', default=False, help='compile with inductor backend')
    parser.add_argument('--profile', action='store_true', default=False, help='profile')
    parser.add_argument('--benchmark', action='store_true', default=False, help='test performance')
    parser.add_argument('--accuracy', action='store_true', default=False, help='test accuracy')
    parser.add_argument('-w', '--warmup_iterations', default=-1, type=int, help='number of warmup iterations to run')
    parser.add_argument('-i', '--iterations', default=-1, type=int, help='number of total iterations to run')
    parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str, help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='ccl', type=str, help='distributed backend')
    parser.add_argument("--weight-sharing", action='store_true', default=False, help="using weight_sharing to test the performance of inference")
    parser.add_argument("--number-instance", default=0, type=int, help="the instance numbers for test the performance of latcy, only works when enable weight-sharing")

    args = parser.parse_args()
    return args

def run_weights_sharing_model(pipe, tid, args):
    total_time = 0
    for i in range(args.iterations + args.warmup_iterations):
        # run model
        start = time.time()
        if args.precision == "bf16" or args.precision == "fp16" or args.precision == "int8-bf16":
            with torch.cpu.amp.autocast(dtype=args.dtype), torch.no_grad():
                output = pipe(args.prompt, generator=torch.manual_seed(args.seed)).images
        else:
            with torch.no_grad():
                output = pipe(args.prompt, generator=torch.manual_seed(args.seed)).images
        end = time.time()
        print('time per prompt(s): {:.2f}'.format((end - start)))
        if i >= args.warmup_iterations:
            total_time += end - start

    print("Instance num: ", tid)
    print("Latency: {:.2f} s".format(total_time / args.iterations))
    print("Throughput: {:.5f} samples/sec".format(args.iterations / total_time))

def main():

    args = parse_args()
    logging.info(f"Parameters {args}")

    # CCL related
    os.environ['MASTER_ADDR'] = str(os.environ.get('MASTER_ADDR', '127.0.0.1'))
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
    os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1))

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
        print("World size: ", args.world_size)

    args.distributed = args.world_size > 1 
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])

    # load model
    pipe = DiffusionPipeline.from_pretrained(args.model_name_or_path)

    # data type
    if args.precision == "fp32":
        print("Running fp32 ...")
        args.dtype=torch.float32
    elif args.precision == "bf32":
        print("Running bf32 ...")
        args.dtype=torch.float32
    elif args.precision == "bf16":
        print("Running bf16 ...")
        args.dtype=torch.bfloat16
    elif args.precision == "fp16":
        print("Running fp16 ...")
        args.dtype=torch.half
    elif args.precision == "int8-bf16":
        print("Running int8-bf16 ...")
        args.dtype=torch.bfloat16
    elif args.precision == "int8-fp32":
        print("Running int8-fp32 ...")
        args.dtype=torch.float32
    else:
        raise ValueError("--precision needs to be the following: fp32, bf32, bf16, fp16, int8-bf16, int8-fp32")

    if args.compile_inductor:
        pipe.precision = torch.float32
    elif args.model_name_or_path == "SimianLuo/LCM_Dreamshaper_v7" and args.precision == "int8-bf16":
        pipe.precision = torch.float32
    else:
        pipe.precision = args.dtype
    if args.model_name_or_path == "stabilityai/stable-diffusion-2-1":
        text_encoder_input = torch.ones((1, 77), dtype=torch.int64)
        input = torch.randn(2, 4, 96, 96).to(memory_format=torch.channels_last).to(dtype=pipe.precision), torch.tensor(921), torch.randn(2, 77, 1024).to(dtype=pipe.precision)
    elif args.model_name_or_path == "SimianLuo/LCM_Dreamshaper_v7":
        text_encoder_input = torch.ones((1, 77), dtype=torch.int64)
        input = torch.randn(1, 4, 96, 96).to(memory_format=torch.channels_last).to(dtype=pipe.precision), torch.tensor(921), torch.randn(1, 77, 768).to(dtype=pipe.precision), torch.randn(1, 256).to(dtype=pipe.precision)
    else:
         raise ValueError("This script currently only supports stabilityai/stable-diffusion-2-1 and SimianLuo/LCM_Dreamshaper_v7.")

    # ipex
    if args.ipex:
        print("Running IPEX ...")
        import intel_extension_for_pytorch as ipex
        if args.precision == "fp32":
            pipe.text_encoder = ipex.optimize(pipe.text_encoder.eval(), inplace=True, sample_input=text_encoder_input)
            pipe.unet = ipex.optimize(pipe.unet.eval(), inplace=True)
            pipe.vae = ipex.optimize(pipe.vae.eval(), inplace=True)
        elif args.precision == "bf32":
            ipex.set_fp32_math_mode(mode=ipex.FP32MathMode.BF32, device="cpu")
            pipe.text_encoder = ipex.optimize(pipe.text_encoder.eval(), inplace=True, auto_kernel_selection=True, sample_input=text_encoder_input)
            pipe.unet = ipex.optimize(pipe.unet.eval(), inplace=True, auto_kernel_selection=True)
            pipe.vae = ipex.optimize(pipe.vae.eval(), inplace=True, auto_kernel_selection=True)
        elif args.precision == "bf16" or args.precision == "fp16":
            # pipe.text_encoder = ipex.optimize(pipe.text_encoder.eval(), dtype=args.dtype, inplace=True, sample_input=text_encoder_input)
            pipe.unet = ipex.optimize(pipe.unet.eval(), dtype=args.dtype, inplace=True)
            pipe.vae = ipex.optimize(pipe.vae.eval(), dtype=args.dtype, inplace=True)
        elif args.precision == "int8-bf16" or args.precision == "int8-fp32":
            if args.model_name_or_path == "stabilityai/stable-diffusion-2-1":
                pipe.text_encoder = ipex.optimize(pipe.text_encoder.eval(), dtype=args.dtype, inplace=True, sample_input=text_encoder_input)
                from quantization_modules import load_int8_model, convert_to_fp32model
                pipe.unet = load_int8_model(pipe.unet, args.int8_model_path)
                pipe.unet = convert_to_fp32model(pipe.unet)
                pipe.vae = ipex.optimize(pipe.vae.eval(), dtype=args.dtype, inplace=True)
            else:
                if not args.calibration:
                    qconfig_mapping = ipex.quantization.default_static_qconfig_mapping
                    pipe.unet = ipex.quantization.prepare(pipe.unet, qconfig_mapping, input, inplace=True)
                    pipe.unet.load_qconf_summary(qconf_summary=args.configure_dir)
                    if args.precision == "int8-fp32":
                        pipe.unet = ipex.quantization.convert(pipe.unet)
                    else:
                        with torch.cpu.amp.autocast():
                            pipe.unet = ipex.quantization.convert(pipe.unet)
                    print("running int8 evalation step\n")
        else:
            raise ValueError("--precision needs to be the following:: fp32, bf32, bf16, fp16, int8-bf16, int8-fp32")

    if args.distributed:
        import oneccl_bindings_for_pytorch
        torch.distributed.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size,
                                rank=args.rank)
        print("Rank and world size: ", torch.distributed.get_rank()," ", torch.distributed.get_world_size())              
        # print("Create DistributedDataParallel in CPU")
        # pipe = torch.nn.parallel.DistributedDataParallel(pipe)

    if args.accuracy:
        # prepare dataloader
        val_coco = dset.CocoCaptions(root = '{}/val2017'.format(args.dataset_path),
                                    annFile = '{}/annotations/captions_val2017.json'.format(args.dataset_path),
                                    transform=transforms.Compose([transforms.Resize((512, 512)), transforms.PILToTensor(), ]))

        if args.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_coco, shuffle=False)
        else:
            val_sampler = None

        val_dataloader = torch.utils.data.DataLoader(val_coco,
                                                    batch_size=1,
                                                    shuffle=False,
                                                    num_workers=0,
                                                    sampler=val_sampler)

    if args.model_name_or_path == "SimianLuo/LCM_Dreamshaper_v7" and args.calibration:
        print("runing LCM int8 calibration step\n")
        qconfig_mapping = ipex.quantization.default_static_qconfig_mapping
        pipe.unet = ipex.quantization.prepare(pipe.unet, qconfig_mapping, input, inplace=True)
        for i, (images, prompts) in enumerate(tqdm(val_dataloader)):
            prompt = prompts[0][0]
            pipe(prompt, generator=torch.manual_seed(args.seed))
            if i == 9:
                break
        pipe.unet.save_qconf_summary(args.configure_dir)
        print(".........calibration step done..........")
        return

    # jit trace
    if args.jit:
        print("JIT trace ...")
        # from utils_vis import make_dot, draw
        if args.precision == "bf16" or args.precision == "fp16":
            with torch.cpu.amp.autocast(dtype=args.dtype), torch.no_grad():
                pipe.traced_unet = torch.jit.trace(pipe.unet, input, strict=False)
                pipe.traced_unet = torch.jit.freeze(pipe.traced_unet)
                pipe.traced_unet(*input)
                pipe.traced_unet(*input)
                # print(pipe.traced_unet.graph_for(input))
        elif args.precision == "int8-bf16":
            with torch.cpu.amp.autocast(), torch.no_grad():
                pipe.traced_unet = torch.jit.trace(pipe.unet, input, strict=False)
                pipe.traced_unet = torch.jit.freeze(pipe.traced_unet)
                pipe.traced_unet(*input)
                pipe.traced_unet(*input)
                # print(pipe.traced_unet.graph_for(input))
        elif args.precision == "int8-fp32":
            with torch.no_grad():
                pipe.traced_unet = torch.jit.trace(pipe.unet, input, strict=False)
                pipe.traced_unet = torch.jit.freeze(pipe.traced_unet)
                pipe.traced_unet(*input)
                pipe.traced_unet(*input)
                # print(pipe.traced_unet.graph_for(input))
        else:
            with torch.no_grad():
                pipe.traced_unet = torch.jit.trace(pipe.unet, input, strict=False)
                pipe.traced_unet = torch.jit.freeze(pipe.traced_unet)
                pipe.traced_unet(*input)
                pipe.traced_unet(*input)
                # print(pipe.traced_unet.graph_for(input))

    # torch.compile with ipex backend
    if args.compile_ipex:
        print("torch.compile with ipex backend ...")
        if args.precision == "fp32" or args.precision == "bf16":
            import intel_extension_for_pytorch as ipex
            pipe.text_encoder = ipex.optimize(pipe.text_encoder.eval(), dtype=args.dtype, weights_prepack=False)
            pipe.unet = ipex.optimize(pipe.unet.eval(), dtype=args.dtype, weights_prepack=False)
            pipe.vae = ipex.optimize(pipe.vae.eval(), dtype=args.dtype, weights_prepack=False)
            ipex._set_compiler_backend("torchscript")
            pipe.text_encoder = torch.compile(pipe.text_encoder, backend="ipex")
            pipe.unet = torch.compile(pipe.unet, backend="ipex")
            pipe.vae.decode = torch.compile(pipe.vae.decode, backend="ipex")
        else:
            raise ValueError("If you want to use torch.compile with ipex backend, --precision needs to be the following: fp32, bf16")
    # torch.compile with inductor backend
    if args.compile_inductor:
        print("torch.compile with inductor backend ...")
        # torch._inductor.config.profiler_mark_wrapper_call = True
        # torch._inductor.config.cpp.enable_kernel_profile = True
        from torch._inductor import config as inductor_config
        inductor_config.cpp_wrapper = True
        if args.precision == "fp32":
            with torch.no_grad():
                pipe.unet = torch.compile(pipe.unet)
                pipe.unet(*input)
                pipe.unet(*input)
                pipe.text_encoder = torch.compile(pipe.text_encoder)
                pipe.vae.decode = torch.compile(pipe.vae.decode)
        elif args.precision == "bf16":
            with torch.cpu.amp.autocast(), torch.no_grad():
                pipe.unet = torch.compile(pipe.unet)
                pipe.unet(*input)
                pipe.unet(*input)
                pipe.text_encoder = torch.compile(pipe.text_encoder)
                pipe.vae.decode = torch.compile(pipe.vae.decode)
        elif args.precision == "fp16":
            torch._C._set_sdp_use_flash(False)
            with torch.cpu.amp.autocast(dtype=torch.half), torch.no_grad():
                pipe.unet = torch.compile(pipe.unet)
                pipe.unet(*input)
                pipe.unet(*input)
                pipe.text_encoder = torch.compile(pipe.text_encoder)
                pipe.vae.decode = torch.compile(pipe.vae.decode)
        elif args.precision == "int8-fp32":
            from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
            import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
            from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
            from torch._export import capture_pre_autograd_graph
            with torch.no_grad():
                pipe.traced_unet = capture_pre_autograd_graph(pipe.unet, input)
                quantizer = X86InductorQuantizer()
                quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())
                pipe.traced_unet = prepare_pt2e(pipe.traced_unet, quantizer)
                # calibration
                pipe(args.prompt)
                pipe.traced_unet = convert_pt2e(pipe.traced_unet)
                torch.ao.quantization.move_exported_model_to_eval(pipe.traced_unet)
                pipe.traced_unet = torch.compile(pipe.traced_unet)
                pipe.traced_unet(*input)
                pipe.traced_unet(*input)
                pipe.text_encoder = torch.compile(pipe.text_encoder)
                pipe.vae.decode = torch.compile(pipe.vae.decode)
        elif args.precision == "int8-bf16":
            from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
            import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
            from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
            from torch._export import capture_pre_autograd_graph
            with torch.no_grad():
                pipe.traced_unet = capture_pre_autograd_graph(pipe.unet, input)
                quantizer = X86InductorQuantizer()
                quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())
                pipe.traced_unet = prepare_pt2e(pipe.traced_unet, quantizer)
                # calibration
                pipe(args.prompt)
                pipe.traced_unet = convert_pt2e(pipe.traced_unet)
                torch.ao.quantization.move_exported_model_to_eval(pipe.traced_unet)
            with torch.cpu.amp.autocast(), torch.no_grad():
                pipe.traced_unet = torch.compile(pipe.traced_unet)
                pipe.traced_unet(*input)
                pipe.traced_unet(*input)
                pipe.text_encoder = torch.compile(pipe.text_encoder)
                pipe.vae.decode = torch.compile(pipe.vae.decode)
        else:
            raise ValueError("If you want to use torch.compile with inductor backend, --precision needs to be the following: fp32, bf16, int8-bf16, int8-fp32")

    # benchmark
    if args.benchmark:
        print("Running benchmark ...")
        if args.weight_sharing:
            print("weight sharing ...")
            threads = []
            for i in range(1, args.number_instance+1):
                thread = threading.Thread(target=run_weights_sharing_model, args=(pipe, i, args))
                threads.append(thread)
                thread.start()
            for thread in threads:
                thread.join()
            exit()
        else:
            total_time = 0
            for i in range(args.iterations + args.warmup_iterations):
                # run model
                start = time.time()
                if args.precision == "bf16" or args.precision == "fp16" or args.precision == "int8-bf16":
                    with torch.cpu.amp.autocast(dtype=args.dtype), torch.no_grad():
                        output = pipe(args.prompt, generator=torch.manual_seed(args.seed)).images
                else:
                    with torch.no_grad():
                        output = pipe(args.prompt, generator=torch.manual_seed(args.seed)).images
                end = time.time()
                print('time per prompt(s): {:.2f}'.format((end - start)))
                if i >= args.warmup_iterations:
                    total_time += end - start

            print("Latency: {:.2f} s".format(total_time / args.iterations))
            print("Throughput: {:.5f} samples/sec".format(args.iterations / total_time))

    if args.accuracy:
        print("Running accuracy ...")
        # run model
        if args.distributed:
            torch.distributed.barrier()
        fid = FrechetInceptionDistance(normalize=True)
        for i, (images, prompts) in enumerate(tqdm(val_dataloader)):
            prompt = prompts[0][0]
            real_image = images[0]
            print("prompt: ", prompt)
            if args.precision == "bf16" or args.precision == "fp16" or args.precision == "int8-bf16":
                with torch.cpu.amp.autocast(dtype=args.dtype), torch.no_grad():
                    output = pipe(prompt, generator=torch.manual_seed(args.seed), output_type="numpy").images
            else:
                with torch.no_grad():
                    output = pipe(prompt, generator=torch.manual_seed(args.seed), output_type="numpy").images

            if args.output_dir:
                if not os.path.exists(args.output_dir):
                    os.mkdir(args.output_dir)
                image_name = time.strftime("%Y%m%d_%H%M%S")
                Image.fromarray((output[0] * 255).round().astype("uint8")).save(f"{args.output_dir}/fake_image_{image_name}.png")
                Image.fromarray(real_image.permute(1, 2, 0).numpy()).save(f"{args.output_dir}/real_image_{image_name}.png")

            fake_image = torch.tensor(output[0]).unsqueeze(0).permute(0, 3, 1, 2)
            real_image = real_image.unsqueeze(0) / 255.0
            
            fid.update(real_image, real=True)
            fid.update(fake_image, real=False)

            if args.iterations > 0 and i == args.iterations - 1:
                break

        print(f"FID: {float(fid.compute())}")

    # profile
    if args.profile:
        print("Running profiling ...")
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as p:
            if args.precision == "bf16" or args.precision == "fp16" or args.precision == "int8-bf16":
                with torch.cpu.amp.autocast(dtype=args.dtype), torch.no_grad():
                    pipe(args.prompt, generator=torch.manual_seed(args.seed)).images
            else:
                with torch.no_grad():
                    pipe(args.prompt, generator=torch.manual_seed(args.seed)).images

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
