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
import io
import os
import json
import sys
import torch
import time

# sample modules
from arguments_utils import args
import loader_utils
import io_utils
import barrier_utils
import statistics_utils

class Inference:
    def __init__(self):
        self.model = None
        self.validation_dataset = None

        # Default settings
        self.use_autocast = False
        self.autocast_dtype = torch.float32

        # Timer variables
        self.start_benchmark_time = None
        self.current_benchmark_time = None
        self.last_status_print_time = None

        # Tracking objects for calculating accuracy
        self.outputs = []

        # Reporting objects
        self.average_meters = {}
        self.progress = None

        # Counter for batches run
        self.actual_batches_run = 0

    def get_device_type(self):
        if args.gpu:
            return 'cuda'
        elif args.xpu:
            return 'xpu'
        return 'cpu'

    def get_model(self):
        if self.model == None:
            io_utils.write_error('Called get model before Inference.load() was called!')
            sys.exit(1)

        return self.model
    
    def get_validation_dataset(self):
        if self.validation_dataset == None:
            io_utils.write_error('Called get validation dataset before Inference.load() was called!')
            sys.exit(1)

        return self.validation_dataset

    def load(self):
        # create model
        self.model = loader_utils.load_model()

        # Data loading code
        self.validation_dataset = loader_utils.load_validation_dataset(
            args.batch_size,
            args.num_inputs,
            args.width,
            args.height,
            loader_utils.validate_data_src(args.data, args.dummy),
            data_workers=4,
            pin_memory_device='xpu' if args.xpu and args.ipex else 'cuda' if args.gpu else None
        )

        # Final config of the model
        self.inference_config()

    def inference_config(self):
        # Ensure model is loaded
        self.get_model()

        # Determine autocast settings
        if not args.no_amp:
            if args.bf16 or args.fp16:
                io_utils.write_info('Using autocast')
                self.use_autocast = True
        else:
            if args.bf16 or args.fp16:
                io_utils.write_info('Not using autocast')
        if args.fp16:
            self.autocast_dtype = torch.float16
        elif args.bf16:
            self.autocast_dtype = torch.bfloat16

        # Mark parameters as non-trainable
        self.model.trainable = False
        io_utils.write_info('Model training disabled')

        # switch to evaluate mode
        self.model.eval()
        io_utils.write_info('Model in eval mode')
        
        # Convert model type if not using float32 and not using autocast
        if args.no_amp:
            if args.fp16:
                self.model = self.model.to(dtype=torch.float16)
                io_utils.write_info('Model converted to FP16')
            elif args.bf16:
                self.model = self.model.to(dtype=torch.bfloat16)
                io_utils.write_info('Model converted to BF16')

        # Set gradients
        if args.no_grad:
            self.model.zero_grad(set_to_none=True)
            io_utils.write_info('Model using no gradient')
        elif args.zero_grad:
            self.model.zero_grad()
            io_utils.write_info('Model using zero gradient')

        # Apply jit conversion if not already done by 1) quantization or 2) loading a saved model
        if args.jit_trace and not args.load:
            trace_input = torch.randn(args.batch_size, 3, args.width, args.height)
            trace_input = self.process_images(trace_input)
            io_utils.write_info('Model using JIT trace')
            with torch.inference_mode():
                if args.xpu and args.ipex:
                    with torch.xpu.amp.autocast(enabled=self.use_autocast, dtype=self.autocast_dtype, cache_enabled=False):
                        self.model = torch.jit.trace(self.model, trace_input)
                elif args.gpu:
                    with torch.autocast(enabled=self.use_autocast, device_type=self.get_device_type(), dtype=self.autocast_dtype, cache_enabled=False):
                        self.model = torch.jit.trace(self.model, trace_input)
        elif args.jit_script and not args.load:
            io_utils.write_info('Model using JIT script')
            self.model = torch.jit.script(self.model)
        elif args.compile and not args.load:
            io_utils.write_info('Model using torch.compile')
            self.model = torch.compile(self.model)


    def process_images(self, images, dtype=None):
        if args.channels_last:
            images = images.to(memory_format=torch.channels_last)
        if args.xpu or args.gpu:
            images = images.to(args.device, non_blocking=args.non_blocking)
        if dtype != None:
            images = images.to(dtype=dtype)
        elif args.no_amp:
            if args.fp16:
                images = images.to(dtype=torch.float16)
            elif args.bf16:
                images = images.to(dtype=torch.bfloat16)

        return images
    
    def synchronize(self):
        if args.xpu:
            torch.xpu.synchronize(args.device)
        elif args.gpu:
            torch.cuda.synchronize(args.device)

    def inference(self, images, target=None):
        if self.start_benchmark_time == None:
            self.start_benchmark_time = time.time()
            self.last_status_print_time = self.start_benchmark_time
        # compute output
        self.outputs += [(self.model(images), target)]
        self.actual_batches_run += 1

    def log_perf(self):
        duration_eval = (self.current_benchmark_time - self.last_status_print_time) / len(self.outputs)
        latency_ms = duration_eval * 1000
        throughput_img_s = args.batch_size / duration_eval

        self.average_meters['latency'].update(
            latency_ms,
            n=len(self.outputs)
            ) # milliseconds
        self.average_meters['throughput'].update(
            throughput_img_s,
            n=len(self.outputs)
            ) # IMG/S
        avg_acc1, avg_acc5 = 0.0, 0.0
        if args.dummy:
            self.average_meters['top1'].update(
                0.0,
                n=len(self.outputs)
            )
            self.average_meters['top5'].update(
                0.0,
                n=len(self.outputs)
            )
        else:
            for output, target in self.outputs:
                # Copy to CPU
                output = output.cpu()

                # Calculate stats
                if target != None:
                    acc1, acc5 = statistics_utils.accuracy(output.float(), target, topk=(1, 5))
                    avg_acc1 += acc1
                    avg_acc5 += acc5

                # record stats
                self.average_meters['top1'].update(acc1[0])
                self.average_meters['top5'].update(acc5[0])
            avg_acc1 /= len(self.outputs)
            avg_acc5 /= len(self.outputs)

        # Calculate how long we have ran the benchmark so far
        current_benchmark_duration = self.current_benchmark_time - self.start_benchmark_time
        # Estimate how long a single pass through the dataset takes
        dataset_time_estimate = current_benchmark_duration * (args.total_batches / self.average_meters['throughput'].count)
        estimated_total_duration = 0
        if args.dummy:
            # Estimated total duration must be a multiple of complete passes through the dataset that exceeds min test duration
            while estimated_total_duration < args.min_test_duration:
                estimated_total_duration += dataset_time_estimate
            # Estimated total duration is hard capped at max test duration
            if estimated_total_duration > args.max_test_duration:
                estimated_total_duration = args.max_test_duration
        else:
            # Total estimate is current latency (in seconds) times number of total batches (latency is per batch).
            estimated_total_duration = (self.average_meters['latency'].avg / 1000) * args.total_batches
        self.progress.display('~{0:5.1f}% (Estimated {1:3.0f}s remaining)'.format(
            min([100, 100 if estimated_total_duration == 0 else 100 * current_benchmark_duration / estimated_total_duration]),
            max([0, estimated_total_duration - current_benchmark_duration])
        ))
        statistics_utils.log_raw_perf_to_file(
            'raw_perf',
            ('throughput', 'images/s', throughput_img_s),
            ('accuracy-top1', '%', avg_acc1),
            ('accuracy-top5', '%', avg_acc5),
            ('latency', 'ms', latency_ms),
            ('batches-per-time-step', 'count', len(self.outputs))
        )

        # Clear output trackers after we have logged their results
        self.outputs = []

        # Update last log time
        self.last_status_print_time = time.time()

    def is_done(self, batch_idx):
        # Update timers
        self.current_benchmark_time = time.time()
        current_benchmark_duration = self.current_benchmark_time - self.start_benchmark_time

        self.benchmark_done = False
        # If we finished a loop through the dataset then we try to be done
        if batch_idx == args.total_batches - 1:
            # If we still have not hit min test duration yet then we must continue testing
            if current_benchmark_duration < args.min_test_duration:
                self.benchmark_done = False
            # We are done
            else:
                self.benchmark_done = True
                return self.benchmark_done

        # Force testing to conclude if we hit max test duration
        if current_benchmark_duration >= args.max_test_duration:
            self.benchmark_done = True

        return self.benchmark_done

    def do_warmup(self):
        # Ensure model is loaded
        self.get_model()

        # Ensure dataset is loaded
        self.get_validation_dataset()

        # Main perf testing loop.
        io_utils.write_info('Starting warmup')
        self.outputs = []
        with torch.inference_mode():
            if args.channels_last:
                io_utils.write_info('Images will be converted to channels last format')
            self.synchronize()
            for batch_index, (images, target) in enumerate(self.validation_dataset):
                if len(self.outputs) >= args.warm_up:
                    break

                images = self.process_images(images)

                if args.jit_trace:
                    self.inference(images, target)
                elif args.xpu and args.ipex:
                    with torch.xpu.amp.autocast(enabled=self.use_autocast, dtype=self.autocast_dtype, cache_enabled=True):
                        self.inference(images, target)
                else:
                    with torch.autocast(enabled=self.use_autocast, device_type=self.get_device_type(), dtype=self.autocast_dtype, cache_enabled=True):
                        self.inference(images, target)

                # synchronize after warmup is complete
                self.synchronize()

        io_utils.write_info('Completed {0} warmup batches'.format(len(self.outputs)))

    def do_perf_benchmarking(self):
        # Ensure model is loaded
        self.get_model()

        # Ensure dataset is loaded
        self.get_validation_dataset()
        args.total_batches = len(self.validation_dataset)

        # Safe the model before doing perf
        if args.save:
            loader_utils.save_model_to_file(self.model)

        # Create progress meter for output to terminal
        self.average_meters = {}
        self.average_meters['throughput'] = statistics_utils.average_meter('Throughput', ':.2f', ' img/s')
        self.average_meters['latency'] = statistics_utils.average_meter('Latency', ':.2f', ' ms')
        self.average_meters['top1'] = statistics_utils.average_meter('Acc@1', ':.2f', '%')
        self.average_meters['top5'] = statistics_utils.average_meter('Acc@5', ':.2f', '%')
        self.progress = statistics_utils.progress_meter(
            None,
            [
                self.average_meters['throughput'],
                self.average_meters['latency'],
                self.average_meters['top1'],
                self.average_meters['top5']
            ],
            prefix='INFO[{0}/{1}]: PERF_STATUS'.format(args.instance, args.total_instances)
        )

        # Reset variables for new benchmarking epoch
        self.start_benchmark_time = None
        self.last_status_print_time = None
        self.outputs = []
        self.benchmark_done = False
        self.actual_batches_run = 0

        # Main perf testing loop.
        barrier_utils.do_ipc_sync(args.barrier, 'start_perf_benchmark', args.terminate_if_sync_fail)
        io_utils.write_info('Starting inference perf testing on {0} unique batches...'.format(args.total_batches))
        if args.dummy:
            io_utils.write_warning('Since dummy data is being used expect accuracy to be zero.')

        with torch.inference_mode():
            if args.channels_last:
                io_utils.write_info('Images will be converted to channels last format')

            if args.dummy:
                # Do preemptive processing of images to save on overhead.
                # NOTE: this increases memory usage which in turn limits dataset
                # size (on large dataset device might run out of resources).
                dataset = []
                for batch_index, (images, target) in enumerate(self.validation_dataset):
                    images = self.process_images(images)
                    dataset += [(images, target)]
            else:
                dataset = self.validation_dataset

            self.synchronize() # Sync once before inference to make sure any ops such as fetching tensors are completed
            while not self.benchmark_done:
                for batch_index, (images, target) in enumerate(dataset):
                    if not args.dummy:
                        images = self.process_images(images)
                    if args.jit_trace:
                        self.inference(images, target)
                    elif args.xpu and args.ipex:
                        with torch.xpu.amp.autocast(enabled=self.use_autocast, dtype=self.autocast_dtype, cache_enabled=True):
                            self.inference(images, target)
                    else:
                        with torch.autocast(enabled=self.use_autocast, device_type=self.get_device_type(), dtype=self.autocast_dtype, cache_enabled=True):
                            self.inference(images, target)

                    # sync for time measurement
                    self.synchronize()
                    # check if we have hit time requirements and log results
                    if self.is_done(batch_index):
                        # Break out of for-loop if we have exceeded max test duration
                        break

                    # check if we should log perf before continuing to next batch
                    if self.current_benchmark_time - self.last_status_print_time >= args.print_frequency:
                        self.log_perf()

        # If we haven't already displayed the latest data
        if len(self.outputs) > 0:
            self.current_benchmark_time = time.time()
            self.log_perf()

        return self.actual_batches_run, self.average_meters['throughput'].avg, self.average_meters['latency'].avg, self.average_meters['top1'].avg, self.average_meters['top5'].avg

