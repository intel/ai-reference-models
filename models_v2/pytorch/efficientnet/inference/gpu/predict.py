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
import torch.multiprocessing as mp
import sys

# sample modules
from inference_utils import Inference
import arguments_utils
import barrier_utils
import configure_utils
import io_utils
import summary_utils
import telemetry

from arguments_utils import args

def predict(instance, input_args):
    # Setup passed in args and make them global
    args = input_args
    args.instance = instance

    # Validate arguments
    if not args.dummy:
        # These arguments are not allowed when running in accuracy check mode.
        args.min_test_duration = 0
        args.max_test_duration = float('inf')
    if args.max_test_duration < args.min_test_duration:
        io_utils.write_error('Requested max test duration ({0}) is less than min test duration ({1})!'.format(args.max_test_duration, args.min_test_duration))
        sys.exit(1)

    # Enumerate list of devices and check compatibility
    configure_utils.enum_device()

    # Enumerate dtypes to be used
    configure_utils.enum_dtypes()

    # Set environment variables to enable optimizations
    configure_utils.set_environ()

    # Apply various device configurations such as auto mixed precision, eager mode, etc.
    configure_utils.config_devices()

    # Show test conditions
    summary_utils.show_test_conditions()

    # Create inference object
    inference = Inference()

    # Load model and dataset
    inference.load()

    # Do warmup on the model and sync processes afterwards
    inference.do_warmup()
    barrier_utils.do_ipc_sync(args.barrier, 'warmup', args.terminate_if_sync_fail)

    # Start benchmarker and sync processes afterwards
    if args.socket != '':
        if args.instance == 1: # Only start a single time
            try:
                telemetry.start(args.socket)
            except Exception as e:
                io_utils.write_error(str(e))
                io_utils.write_error('failed to communicate with telemetry server')
                sys.exit(1)
        barrier_utils.do_ipc_sync(args.barrier, 'telemetry', args.terminate_if_sync_fail)

    # Do inference benchmarking
    batches_tested, throughput, latency, top1, top5 = inference.do_perf_benchmarking()

    # Stop benchmarker
    if args.socket != '':
        if args.instance == 1: # Only stop a single time
            try:
                telemetry.stop(args.socket)
            except Exception as e:
                io_utils.write_error(str(e))
                io_utils.write_error('failed to communicate with telemetry server')
                sys.exit(1)

    # Write process specific results
    summary_utils.write_results(batches_tested, throughput, latency, top1, top5)

def main():
    arguments_utils.parse_arguments()

    # Set sharing to avoid 'too many open files' error
    # See: https://github.com/pytorch/pytorch/issues/11201
    mp.set_sharing_strategy('file_system')
    args.barrier = mp.Barrier(args.total_instances, timeout=args.max_wait_for_sync)
    try:
        mp.set_start_method('spawn')
    except:
        pass
    processes = {}
    for instance in range(1, args.total_instances + 1):
        processes[instance] = mp.Process(target=predict, args=(instance, args))
    for instance in processes:
        processes[instance].start()
    has_error = False
    for instance in processes:
        processes[instance].join()
        if processes[instance].exitcode != 0:
            io_utils.write_error('Process {0} returned non-zero exit code: {1}'.format(instance, processes[instance].exitcode))
            has_error = True
    if has_error:
        sys.exit(1)

    summary_utils.combine_results()
    sys.exit(0)

if __name__ == '__main__':
    main()
