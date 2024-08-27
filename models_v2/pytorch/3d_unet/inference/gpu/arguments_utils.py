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
    parser = argparse.ArgumentParser(description='PyTorch 3D-UNet inference')
    parser.add_argument('--data', metavar='DIR', nargs='?', default="build/preprocessed_data",
                        help="path to preprocessed data (3D-UNet dataset must be preprocessed see README)")
    parser.add_argument("--label-data-dir", default="build/raw_data/nnUNet_raw_data/Task043_BraTS2019/labelsTr",
                        help="Path to the directory containing ground truth labels")
    parser.add_argument('--output-dir', default='/workspace/temp/',
                        help='path to output results (default: /workspace/temp/)')
    parser.add_argument('-a', '--arch', metavar='ARCH', 
                        default="build/result/nnUNet/3d_fullres/Task043_BraTS2019/nnUNetTrainerV2__nnUNetPlansv2.mlperf.1",
                        help="Path to the directory containing plans.pkl")
    parser.add_argument('-b', '--batch-size', default=1, type=int,
                        metavar='N',
                        help='mini-batch size (default: 1), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--batch-streaming', default=1, type=int,
                        metavar='N',
                        help='Aggregate data over this number of batches before calculating stats')
    parser.add_argument('--max-val-dataset-size', default=1, type=int,
                        metavar='N',
                        help='limit number of images to use for validation')
    parser.add_argument('--status-prints', default=10, type=int,
                        metavar='N', help='number of status prints during benchmarking (default: 10)')
    parser.add_argument('--device', default='cpu', type=str,
                        help='device to use (cpu, cuda[:n], xpu[:n], etc.)')
    parser.add_argument('--ipex', action='store_true',
                        help='use Intel Extension for PyTorch for xpu device')
    parser.add_argument('--tf32', default=0, type=int, help='Datatype used: TF32')
    parser.add_argument('--bf32', default=0, type=int, help='Datatype used: BF32')
    parser.add_argument('--fp16', default=0, type=int, help='Datatype used: FP16')
    parser.add_argument('--bf16', default=0, type=int, help='Datatype used: BF16')
    parser.add_argument('--int8', default=0, type=int, help='Use signed int8 quantization to do inference')
    parser.add_argument('--uint8', default=0, type=int, help='Use unsigned int8 quantization to do inference')
    parser.add_argument('--asymmetric-quantization', dest='asymmetric_quantization', action='store_true',
                        help='Enable asymmetric quantization (default is symmetric).')
    parser.add_argument('--use-jit', type= str, choices=['none', 'trace', 'script', 'compile'], default='trace', required=False,
                        help='enable JIT trace/script/compile graph mode (Default=trace)')
    parser.add_argument('--no-amp', action='store_true',
                        help='Do not use autocast. Direct conversion from native data type to desired data type')
    parser.add_argument('--calib-iters', default=8, type=int,
                        help='iteration number for calibration')
    parser.add_argument('--calib-bs', default=1, type=int,
                        metavar='N', help='mini-batch size for calibration')
    parser.add_argument('--perchannel-weight', default=False,
                        help='do calibration with weight per channel quantization')
    parser.add_argument('--non-blocking', default=False, action='store_true',
                        help='non blocking H2D for input and target, default False')
    parser.add_argument('--sync-after-batch', action='store_true', help='synchronize after each batch and perform accuracy check on model output')
    parser.add_argument('--channels-last', action='store_true', help='enable channels last')
    parser.add_argument('--warm-up', default=10, type=int, help='warm up batches')
    parser.add_argument('--dummy', action='store_true', help='use dummy data for '
                        'benchmark training or val')
    parser.add_argument('--total-instances', default=1, type=int, help='total number of instances for multi-process inference perf measurement')
    parser.add_argument('--max-wait-for-sync', default=60, type=int, help='max amount of time that will be spend waiting for sync success')
    parser.add_argument('--terminate-if-sync-fail', action='store_true', help='terminate process if fail to sync')

    global args
    parser.parse_args(namespace=args)
