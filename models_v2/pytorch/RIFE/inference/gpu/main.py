# Copyright (c) 2023-2024 Intel Corporation
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

# system modules
import os
import multiprocessing as mp
import sys
import torch
from packaging import version
from pathlib import Path

# sample modules
import rife_util
import barrier_utils
import telemetry
import io_utils
from rife_util import RIFE_Interpolation
from report_utils import write_instance_result, get_env_report, combine_results

try:
    from model.warplayer import warp
    from model.IFNet import IFNet
except:
    raise ImportError("Failed to import RIFE model.\n"
                      "Pls set PYTHONPATH env variable to include path to the model.")

def load_model(args):
    # Load RIFE's FlowNet (IFNet) model, with specified pretrained weights`
    model = IFNet()
    if args.device.startswith('xpu') and not args.ipex:
        # map_location=torch.device('xpu') does not currently work on native XPU backend
        # using torch.device('cpu') as workaround
        pretrained_flownet = torch.load(args.pretrained_weights, map_location=torch.device('cpu'))
    else:
        pretrained_flownet = torch.load(args.pretrained_weights, map_location=args.device)
    new_pretrained_flownet = {
        k.replace("module.", ""): v for k, v in pretrained_flownet.items() if "module." in k
    }
    model.load_state_dict(new_pretrained_flownet)

    # Model evaluation with :func `torch.nn.Module.eval`
    model.eval()

    # Model: Adjust Precision and Load to device
    dtype, _ = rife_util.get_data_type(args.precision, args.amp)
    if not args.amp:
        model = model.to(dtype)
    model = model.to(args.device)
    io_utils.write_info("Loaded ArXiv RIFE model.")

    # Using `optimize` provided by Intel® Extension for PyTorch* for optimization on Intel GPU
    if args.device.startswith('xpu') and args.ipex:
        import intel_extension_for_pytorch as ipex
        model = ipex.optimize(model)

    return model

def process_rife_interpolation(args, instance):
    args.instance = instance
    rife_util.args = args
    # Load IPEX if it is requested. Note that this function is called
    # in a separate process and we need to load explicitly those modules
    # which are not loaded at global scope.
    if args.device.startswith('xpu') and args.ipex:
        import intel_extension_for_pytorch as ipex

    model = load_model(args)
    interp_handle = rife_util.RIFE_Interpolation(model, args, "RIFE", instance)

    # Warm Up with random data
    io_utils.write_info(f"Warming up model with #{args.warmup} Frames")
    interp_handle.run_interpolation(warmup=True)
    warmup_perf_report = { "warmup_result" : interp_handle.pm_global.summarize_perf() }
    interp_handle.pm_global.reset()
    io_utils.write_info("End of warm up phase")
    barrier_utils.do_ipc_sync(args.barrier, 'warmup', terminate_if_sync_fail=True)

    # Start telemetry process if enabled
    if args.socket != '':
        if instance == 1: # Only start a single time
            try:
                telemetry.start(args.socket)
            except Exception as e:
                io_utils.write_error(str(e))
                io_utils.write_error("Failed to communicate with telemetry server")
                sys.exit(1)
        barrier_utils.do_ipc_sync(args.barrier, 'telemetry', terminate_if_sync_fail=True)

    processed_frames = interp_handle.run_interpolation(warmup=False, enable_save=args.saveimages)
    io_utils.write_info(f"Processed {processed_frames} frames")

    # Stop telemetry process
    if args.socket != '' and instance == 1: # Only stop a single time
        try:
            telemetry.stop(args.socket)
        except Exception as e:
            io_utils.write_error(str(e))
            io_utils.write_error("Failed to communicate with telemetry server")
            sys.exit(1)

    # Generate result summaries
    quality_report = interp_handle.get_quality_report()
    perf_report = interp_handle.pm_global.summarize_perf(warmup_perf_report)
    env_report = get_env_report(args, model)
    write_instance_result(args, [warmup_perf_report, quality_report, perf_report,  env_report],
                          tag = instance)

def main():
    #########################
    # Inputs
    # ----------------------------------
    # Parse command-line arguments
    rife_util.parse_arguments()
    rife_util.show_arguments()
    args = rife_util.args

    # Create output directory for test-run
    try:
        os.makedirs(args.output, exist_ok=True)
    except:
        io_utils.write_error(f"Failed to create directory to capture outputs at path: {args.output}")
        raise Exception(f"Valid Output directory needs to be provided with --output-dir option")

    #########################
    # Device Availability
    # ----------------------------------
    # Using `torch.xpu.is_available()` supported by Intel® Extension for PyTorch*,
    # the presence of an Intel GPU device can be checked and fallback is CPU.
    if args.device.startswith('xpu'):
        available = False
        if not args.ipex:
            if version.parse(torch.__version__).release < version.parse('2.4').release:
                io_utils.write_warning(f"XPU backend not available in this PyTorch version (requires v2.4 or later): " + torch.__version__)
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                available = True
        else:
            try:
                import intel_extension_for_pytorch as ipex
                if ipex.xpu.is_available():
                    available = True
            except Exception as e:
                io_utils.write_warning(f"Intel Extension for PyTorch (IPEX) not available: " + str(e))
                pass
        if not available:
            io_utils.write_error(f"XPU backend not available.")
            sys.exit(1)

        device = args.device
        io_utils.write_info("Use XPU: " + device)
        io_utils.write_info("Use IPEX: " + str(args.ipex))
    elif "cuda" in args.device:
        if not hasattr(torch, "cuda") or not torch.cuda.is_available():
            io_utils.write_error(f"Make sure cuda is enabled in torch.")
            sys.exit(1)
        device = args.device
        io_utils.write_info("Use GPU: " + device)
    elif "cpu" == args.device:
        device = "cpu"
        io_utils.write_info("Use CPU")
    else:
        raise ValueError(f"Unknown acceleration type - {args.device}")

    #################################################################
    # Spawn processes and run inference for requested instance count
    # ---------------------------------------------------------------

    # Start method for multi-processing needs to be set to "spawn"
    # This is required to support multi-processing for CUDA
    # For more details, see https://pytorch.org/docs/stable/notes/multiprocessing.html#cuda-in-multiprocessing
    try:
        mp.set_start_method("spawn")
    except Exception as e:
        io_utils.write_warning(str(e))
        pass

    # Create barrier to sync start of processes
    # Note that this needs to happen after set_start_method call. Calling this before
    # seems to cause the start method to set to alternate methods.
    args.barrier = mp.Barrier(args.streams)

    # Create processes
    processes = {}
    for instance in range(1, args.streams + 1):
        processes[instance] = mp.Process(target=process_rife_interpolation, args=(args, instance))

    # Synchronize start of all processes
    for instance in processes:
        processes[instance].start()

    # Wait for processes to finish
    has_error = False
    for instance in processes:
        processes[instance].join()
        if processes[instance].exitcode != 0:
            has_error = True
            io_utils.write_error(f"[P{instance}]: Returned non-zero exit code: {processes[instance].exitcode}")

    combine_results(args)
    return 1 if has_error else 0

if __name__ == "__main__":
    main()
