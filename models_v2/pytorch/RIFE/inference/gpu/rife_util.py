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
import torch
import argparse
from pathlib import Path
from PIL import Image
import sys
import numpy as np
from torch.utils.data import DataLoader
from dataset_utils import Vimeo90K_Test_Dataset, quality_check
from report_utils import PerfManager

# sample modules
import io_utils

args = argparse.Namespace()
default_output_path = os.path.abspath(os.path.join(str(Path(__file__).resolve().parent), "output-dir"))
default_weights_path = os.path.abspath(os.path.join(str(Path(__file__).resolve().parent),
                                                    "train_log/flownet.pkl"))

def parse_arguments():
    global args
    parser = argparse.ArgumentParser(description="PyTorch RIFE inference")
    parser.add_argument("--dummy", action='store_true',
                        help="Skips data generation, and repeats the same data over all inference submissions")
    parser.add_argument("--datadim", dest="ddim", nargs=3, default=[3,720,1280], type=int,
                        help="Dimensions of source tensor if using dummy mode."
                        "Expected to be a 3D input. Used only if --dummy is specifed. ")
    parser.add_argument("--data", metavar='DIR', nargs='?', default=None,
                        help="path to dataset (default: None)")
    parser.add_argument("--output-dir", default=default_output_path, dest="output",
                        help="path to output results")
    parser.add_argument("--interpolation",default=1, type=int,
                        help="number of output frames generated by the inference.\n")
    parser.add_argument("--pretrained-weights", default=default_weights_path,
                        help="Path for weights to be used in the inference model")
    parser.add_argument("--num-inputs", dest="frames", default=100, type=int,
                        help="Number of pairs of input frames to process in the current run."
                        " In --dummy mode, this is overridden by --min/max-test-duration settings."
                        " In --data mode, this is limited by the number of dataset entries.")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Number of frames to run warmup")
    parser.add_argument("--saveimages", action="store_true",
                        help="Saves Output Images")
    parser.add_argument("--device", default="xpu",
                        help="What device to use for processing")
    parser.add_argument('--ipex', action='store_true',
                        help='use Intel Extension for PyTorch for xpu device')
    parser.add_argument("--psnr-threshold", default=25, type=int,
                        help="PSNR Threshold in dB to determine if a test passed")
    parser.add_argument("--min-pass-pct", default=95, type=int,
                        help="Minimum % of passed tests(frames) needed to declare a pass result for the full run. "
                        "Does not apply when --dummy is enabled")
    parser.add_argument('--print-frequency', default=1, type=int,
                        help="Specify interval to print stats in seconds: default 1s")
    parser.add_argument('--min-test-duration', default=0, type=int,
                        help="Minimum interval to run benchmark tests for. Only applicable when --dummy is enabled."
                        "In dummy mode, overides frame limit specified by --num-inputs")
    parser.add_argument('--max-test-duration', default=np.inf, type=int,
                        help="Maximum interval to run benchmark tests for. Only applicable when --dummy is enabled."
                        "In dummy mode, overides frame limit specified by --num-inputs")
    parser.add_argument('--async-count', '--async', default=0, type=int, dest='async_count',
                        help="Number of frames after when a sync is issued for non-CPU device workloads."
                        "Default = 0. Taken to mean a sync is issued only at the end of all submissions.")
    parser.add_argument("--streams", default=1, type=int,
                        help="Total number of instances for multi-process execution")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "fp32"],
                        help="Precision to use to run inference")
    parser.add_argument("--amp", action='store_true',
                        help="Use torch.amp to enable automatic mixed precision package")
    parser.add_argument('--socket', default='', type=str,
                        help="socket to control telemetry capture (default: '' : meaning disabled)")
    parser.parse_args(namespace=args)
    if not validate_args():
        io_utils.write_error('Error in validating args. Aborting run')
        sys.exit(1)

def show_arguments():
    io_utils.write_info('------------ start inference arguments --------------------------')
    io_utils.stdout_helper('  [MODEL]')
    io_utils.stdout_helper('    running model:      RIFE')
    io_utils.stdout_helper('    weights path:       {0}'.format(args.pretrained_weights))
    io_utils.stdout_helper('    device:             {0}'.format(args.device))
    if args.device.startswith('xpu'):
        io_utils.stdout_helper('    using IPEX:         {0}'.format(args.ipex))
    io_utils.stdout_helper('    interpolation:      {0}'.format(args.interpolation))
    io_utils.stdout_helper('  [PASS-FAIL CRITERIA]')
    io_utils.stdout_helper('    PSNR threshold:     {0}'.format(args.psnr_threshold))
    io_utils.stdout_helper('    min pass pct:       {0}'.format(args.min_pass_pct))
    io_utils.stdout_helper('  [INPUT-OUTPUT]')
    io_utils.stdout_helper('    output dir:         {0}'.format(args.output))
    io_utils.stdout_helper('    save images:        {0}'.format(args.saveimages))
    io_utils.stdout_helper('    using dummy data:   {0}'.format(args.dummy))
    if args.dummy:
        io_utils.stdout_helper('    input size:         {0}x{1} {2}-channel input'.format(args.ddim[2], args.ddim[1], args.ddim[0]))
    else:
        io_utils.stdout_helper('    val data dir:       {0}'.format(args.data))
    if args.socket:
        io_utils.stdout_helper('    socket:             {0}'.format(args.socket))
    io_utils.stdout_helper('  [DATA TYPE]')
    io_utils.stdout_helper('    using dtype:        {0}'.format(args.precision))
    if args.precision not in ['fp32']:
        io_utils.stdout_helper('    using amp:          {0}'.format(args.amp))
    io_utils.stdout_helper('  [PERF ARGS]')
    io_utils.stdout_helper('    JIT method:         {0}'.format('none'))
    io_utils.stdout_helper('    async count:        {0}'.format(args.async_count))
    io_utils.stdout_helper('  [BENCHMARK PARAMS]')
    io_utils.stdout_helper('    warm up frames :    {0}'.format(args.warmup))
    io_utils.stdout_helper('    batch size:         {0}'.format(1))
    if args.dummy:
        io_utils.stdout_helper('    min test duration:  {0}'.format(args.min_test_duration))
        io_utils.stdout_helper('    max test duration:  {0}'.format(args.max_test_duration))
    else:
        io_utils.stdout_helper('    num frames:         {0}'.format(args.frames))
    io_utils.stdout_helper('    stream count:       {0}'.format(args.streams))
    io_utils.stdout_helper('  [MISC]')
    io_utils.stdout_helper('    print frequency:    {0}'.format(args.print_frequency))
    io_utils.stdout_helper(' --------------------------- end inference arguments ---------------------------')

def validate_args():
    status = True
    if args.dummy and args.min_test_duration > args.max_test_duration:
        io_utils.write_error("max_test_duration needs to be >= min_test_duration")
        status = False
    if args.dummy and args.data is not None:
        io_utils.write_error("Cannot specify a dataset option with --dummy enabled")
        status = False
    if args.interpolation <= 0:
        io_utils.write_error("interpolation needs to be a positive integer")
        status = False
    return status

def get_data_type(pstr, amp=False):
    dtype = torch.float16 if pstr== "fp16" else torch.bfloat16 if pstr == "bf16" else torch.float32
    use_autocast = False if dtype == torch.float32 or not amp else True
    return dtype, use_autocast

def model_precision_match(model, precision):
    model_params_match = [
        True if param.dtype == precision else False for param in model.parameters()
    ]
    all_params_matched = True if model_params_match and all(model_params_match) else False
    return all_params_matched

class RIFE_Interpolation():
    def __init__(self, model, args, model_name="", instance=0):
        self.device = args.device

        self.model = model
        self.model_name = model_name
        self.instance = instance
        self.frame_count = args.frames
        self.dummy = args.dummy
        self.ddim = tuple(args.ddim)
        self.interpolation = args.interpolation
        self.args = args

        self.precision, self.use_autocast = get_data_type(args.precision, args.amp)
        if not self.use_autocast and not model_precision_match(self.model, self.precision):
            io_utils.write_warning(f'Model Precision not matched with specified {self.precision} precision')

        # Performance and test-control related parameters
        self.pm = PerfManager(self.args)        #Timer to print stats
        self.pm_global = PerfManager(self.args) #Timer to control test-end
        self.report_string = "{process_string}async={async_count}".format(
            process_string="instance = " + str(instance) + ", " ,
            async_count=self.args.async_count)
        self.first_input = None

        # Quality related parameters
        self.test_count = 0
        self.test_fail = 0
        self.test_pass = 0
        self.psnr_threshold = args.psnr_threshold
        self.saveimages = args.saveimages
        self.output = args.output if args.streams == 1 else os.path.join(args.output, str(self.instance))
        if self.saveimages:
            try:
                os.makedirs(self.output, exist_ok=True)
            except:
                io_utils.write_error(f"Failed to create directory to capture images for this process at path: {self.output}")
                sys.exit(1)

        self.data = args.data
        self.result_list = []
        if not self.dummy:
            if not self.data:
                io_utils.write_error(f"Dataset path needs to be provided with --data option")
                sys.exit(1)
            if not os.path.exists(self.data):
                io_utils.write_error(f"Error in finding Dataset path: " + self.data)
                sys.exit(1)
            if self.interpolation > 1:
                io_utils.write_warning(f"Quality check is skipped if interpolation > 1")
            self.test_dataset = Vimeo90K_Test_Dataset(self.data)
            self.test_dl = DataLoader(self.test_dataset, batch_size=1)
            self.test_dl_iter = iter(self.test_dl)

    def preprocess(self, input_tensor):
        # Expects input tensor as NCHW integers and returns NCHW tensor with requested precision on specified device
        # The conversion from int to float is first done with an fp32 compute and then switched to required precision
        processed_tensor = input_tensor.to(torch.float32) / 255.0
        processed_tensor = processed_tensor.to(self.precision).to(self.device)
        return processed_tensor

    def fetch_tensors(self):
        if self.dummy and self.first_input:
            return self.first_input
        if not self.dummy:
            frame1, frame2, frame3  = next(self.test_dl_iter)
        else:
            tensor_dim  = self.ddim
            tensor_type = torch.uint8
            frame1 = torch.randint(size=tensor_dim, low=0, high=256, dtype=tensor_type).unsqueeze(0)
            frame2 = None
            frame3 = torch.randint(size=tensor_dim, low=0, high=256, dtype=tensor_type).unsqueeze(0)
        processed_tensor1 = self.preprocess(frame1)
        processed_tensor2 = self.preprocess(frame2) if (frame2 != None) else None
        processed_tensor3 = self.preprocess(frame3)
        if self.dummy:
            self.first_input = processed_tensor1, processed_tensor2, processed_tensor3
        return processed_tensor1, processed_tensor2, processed_tensor3

    def _synchronize(self):
        if "xpu" in self.device:
            torch.xpu.synchronize()
        elif "cuda" in self.device:
            torch.cuda.synchronize()

    def inference(self, img0, img1, scale=1.0):
        imgs = torch.cat((img0, img1), 1)
        scale_list = [4 / scale, 2 / scale, 1 / scale]
        with torch.no_grad():
            with torch.autocast(enabled=self.use_autocast, device_type=self.device, dtype=self.precision, cache_enabled=True):
                flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.model(imgs, scale_list)
        return merged[2]

    def update_quality_stats(self, qcheck_pass):
        self.test_count += 1
        if qcheck_pass:
            self.test_pass += 1
        else:
            self.test_fail += 1

    def process_result(self, imgt_pred_np, ref_output_np, frame_index=0, save_images=False):
        imgt_pred_np = imgt_pred_np.to("cpu")
        ref_output_np = ref_output_np.to("cpu")
        qcheck_result = True
        for i in range(0, self.interpolation):
            if self.interpolation == 1:
                res = quality_check(imgt_pred_np[i], ref_output_np[i], self.psnr_threshold)
                qcheck_result = False if res == False else qcheck_result
            if save_images:
                processed_imgt_pred = (imgt_pred_np[i].data.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                img = Image.fromarray(processed_imgt_pred)
                img_path = os.path.join(self.output, f"output_frame_{frame_index}_{i}.png")
                img.save(img_path)
        self.update_quality_stats(qcheck_result)

    def check_testend_status(self, frame_index, frame_count, warmup=False):
        # Decide if test needs to be ended
        end_test =  False if frame_index < frame_count else True
        if self.dummy and not warmup:
            end_test = False if not self.pm_global.check_timer_elapsed(self.args.min_test_duration) else end_test
            end_test = True if self.pm_global.check_timer_elapsed(self.args.max_test_duration) else end_test

        # Print status at specified interval
        if (frame_index != 0 and self.pm.check_timer_elapsed(self.args.print_frequency)) or end_test:
            self.pm.end_timer()
            report = self.pm.gather_metrics(frame_index, mode="skip", quality_metrics=self.get_quality_report())
            self.pm.restart_timer()
            self.pm_global.report_metrics(report, frame_count=frame_count, mode="update", display=self.report_string)

        return end_test

    def is_sync(self, frame_index):
        return True if self.args.async_count and (frame_index % self.args.async_count == 0) else False

    def run_interpolation(self, warmup=False, enable_save=False):
        interpolation = self.interpolation
        frame_count = self.args.warmup if warmup else self.args.frames
        save_images = False if (warmup or self.dummy) else enable_save
        frame_index = 0

        if self.dummy:
            try:
                tensor_f1, tensor_f2, tensor_f3 = self.fetch_tensors()
            except:
                io_utils.write_error(f"Failed to get two valid tensors to run RIFE model")
                sys.exit(1)
            frames = [tensor_f1, tensor_f3]
            self._synchronize()
        else:
            # Reset data-loader to start fresh
            self.test_dl_iter = iter(self.test_dl)
            test_dl_len = len(self.test_dl_iter)
            if test_dl_len < frame_count:
                io_utils.write_warning('Number of items in dataset is less than the requested number of inputs')
                io_utils.write_warning('Number of inputs has been changed from {0} to {1}'.format(frame_count, test_dl_len))
                frame_count = test_dl_len

        self.pm_global.start_timer()
        self.pm.start_timer()
        while self.check_testend_status(frame_index, frame_count, warmup) != True:
            # Get next batch of inputs
            if not self.dummy:
                try:
                    tensor_f1, tensor_f2, tensor_f3 = self.fetch_tensors()
                except StopIteration:
                    break

            # Submit input tensors for inference
            imgt_pred_np = self.inference(tensor_f1, tensor_f3)
            frame_index += 1

            if not self.dummy and not warmup:
                self.result_list.append((imgt_pred_np, tensor_f2, frame_index))
                if self.is_sync(frame_index):
                    self._synchronize()
                    for out_image, ref_image, out_index in self.result_list:
                        self.process_result(out_image, ref_image, out_index, save_images)
                    self.result_list = []
            elif self.is_sync(frame_index):
                self._synchronize()

        #Sync after critical loop
        self._synchronize()
        if not self.dummy and not warmup:
            for out_image, ref_image, out_index in self.result_list:
                self.process_result(out_image, ref_image, out_index, save_images)
            self.result_list=[]

        self.pm_global.end_timer()
        self.pm_global.gather_metrics(frame_index, mode="new", display=self.report_string + ", Post Final Sync", quality_metrics=self.get_quality_report())
        return frame_index

    def get_quality_report(self):
        pass_pct = 0 if self.test_count == 0 else 100*self.test_pass/self.test_count
        fail_pct = 0 if self.test_count == 0 else 100*self.test_fail/self.test_count
        decision = True if (self.dummy or pass_pct >= self.args.min_pass_pct) else False
        quality_report = {
            "quality_data" : {
                "test_count" : self.test_count,
                "test_pass"  : self.test_pass,
                "test_fail"  : self.test_fail,
                "test_pass_pct" : pass_pct,
                "test_fail_pct" : fail_pct,
                "decision_pass" : decision,
            },
            "results": {
                "metrics" : {
                    "accuracy" : {
                        "avg" : pass_pct,
                        "min" : pass_pct,
                        "max" : pass_pct,
                        "stdev" : 0.0,
                        "units" : "%"
                    },
                }
            }
        }
        return quality_report
