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
import argparse

args = argparse.Namespace()

def parse_arguments():
    model_names = ['fbnetc_100']

    parser = argparse.ArgumentParser(description='PyTorch FBNet inference')
    parser.add_argument('--data', metavar='DIR', nargs='?', default=None,
                        help='path to dataset (default: None)')
    parser.add_argument('--output-dir', default='/workspace/temp/',
                        help='path to output results (default: /workspace/temp/)')
    parser.add_argument('--socket', default='', type=str,
                        help='socket to control telemetry capture (default: disabled)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='fbnetc_100',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: fbnetc_100)')
    parser.add_argument('--width', default=380, type=int, metavar='N',
                        help='width of input image to be used (default: 380)')
    parser.add_argument('--height', default=380, type=int, metavar='N',
                        help='height of input image to be used (default: 380)')
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                        metavar='N',
                        help='mini-batch size (default: 16), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--num-inputs', default=50000, type=int,
                        metavar='N',
                        help='limit number of images to use for validation (default: 50000)')
    parser.add_argument('--print-frequency', default=1, type=int,
                        metavar='N', help='specify perf stats printing frequency (in seconds) here (default: 1)')
    parser.add_argument('--min-test-duration', default=0.0, type=float,
                        metavar='N', help='minimum length of benchmark in seconds. Will repeat batches until reached. (default: 0 seconds)')
    parser.add_argument('--max-test-duration', default=float('inf'), type=float,
                        metavar='N', help='maximum length of benchmark in seconds. Will terminate benchmark once reached. (default: inf)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--device', default='cpu', type=str,
                        help='device to use (cpu, cuda[:n], xpu[:n], etc.)')
    parser.add_argument('--ipex', action='store_true',
                        help='use Intel Extension for PyTorch for xpu device')
    parser.add_argument('--tf32', default=0, type=int, help='Datatype used: TF32')
    parser.add_argument('--bf32', default=0, type=int, help='Datatype used: BF32')
    parser.add_argument('--fp16', default=0, type=int, help='Datatype used: FP16')
    parser.add_argument('--bf16', default=0, type=int, help='Datatype used: BF16')
    parser.add_argument('--jit-trace', action='store_true',
                        help='enable PyTorch JIT trace graph mode')
    parser.add_argument('--jit-script', action='store_true',
                        help='enable PyTorch JIT script graph mode')
    parser.add_argument('--compile', action='store_true',
                        help='enable PyTorch compile')
    parser.add_argument('--zero-grad', action='store_true',
                        help='set model gradients to zero')
    parser.add_argument('--no-grad', action='store_true',
                        help='set model gradients to none')
    parser.add_argument('--no-amp', action='store_true',
                        help='Do not use autocast. Direct conversion from native data type to desired data type')
    parser.add_argument('--non-blocking', default=False, action='store_true',
                        help='non blocking H2D for input and target, default False')
    parser.add_argument('--channels-last', action='store_true', help='enable channels last')
    parser.add_argument('--warm-up', default=0, type=int, help='warm up batches')
    parser.add_argument('--label-smoothing', default=0.0, type=float)
    parser.add_argument('--dummy', action='store_true', help='use dummy data for '
                        'benchmark training or val')
    parser.add_argument('--save', help='Path to save entire model, save inference mode, training is not available')
    parser.add_argument('--load', help='Path to load entire inference model')
    parser.add_argument('--total-instances', default=1, type=int, help='total number of instances for multi-process inference perf measurement')
    parser.add_argument('--max-wait-for-sync', default=60, type=int, help='max amount of time that will be spend waiting for sync success')
    parser.add_argument('--terminate-if-sync-fail', action='store_true', help='terminate process if fail to sync')

    global args
    parser.parse_args(namespace=args)
