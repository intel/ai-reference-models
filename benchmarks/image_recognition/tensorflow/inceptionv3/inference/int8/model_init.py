#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Intel Corporation
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os


class ModelInitializer:
    """Model initializer for inception v3 int8 inference"""

    args = None
    custom_args = []

    def __init__(self, args, custom_args=[], platform_util=None):
        self.args = args
        self.custom_args = custom_args
        if not platform_util:
            raise ValueError("Did not find any platform info.")

        if self.args.verbose:
            print('Received these standard args: {}'.format(self.args))
            print('Received these custom args: {}'.format(self.custom_args))

        # Environment variables
        os.environ["OMP_NUM_THREADS"] = \
            str(platform_util.num_cores_per_socket()) \
            if self.args.num_cores == -1 else str(self.args.num_cores)
        os.environ["KMP_BLOCKTIME"] = "0"
        os.environ["KMP_SETTINGS"] = "1"
        os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"

    def parse_args(self):
        if self.custom_args:
            parser = argparse.ArgumentParser()
            parser.add_argument(
                "--input-height", default=None,
                dest='input_height', type=int, help="input height")
            parser.add_argument(
                "--input-width", default=None,
                dest='input_width', type=int, help="input width")
            parser.add_argument(
                '--warmup-steps', dest='warmup_steps',
                help='number of warmup steps',
                type=int, default=10)
            parser.add_argument(
                '--steps', dest='steps',
                help='number of steps',
                type=int, default=50)
            parser.add_argument(
                '--num-inter-threads', dest='num_inter_threads',
                help='number threads across operators',
                type=int, default=1)
            parser.add_argument(
                '--num-intra-threads', dest='num_intra_threads',
                help='number threads for an operator',
                type=int, default=1)
            parser.add_argument(
                '--input-layer', dest='input_layer',
                help='name of input layer',
                type=str, default=None)
            parser.add_argument(
                '--output-layer', dest='output_layer',
                help='name of output layer',
                type=str, default=None)

            self.args = parser.parse_args(self.custom_args,
                                          namespace=self.args)

    def run_benchmark(self):
        benchmark_script = os.path.join(self.args.intelai_models,
                                        "benchmark.py")
        script_args_list = [
            "input_graph", "input_height", "input_width", "batch_size",
            "input_layer", "output_layer", "num_inter_threads",
            "num_intra_threads", "warmup_steps", "steps"]

        cmd = self.add_numactl_and_args(benchmark_script, script_args_list)
        os.system(cmd)

    def run_accuracy(self):
        accuracy_script = os.path.join(self.args.intelai_models, "accuracy.py")
        script_args_list = [
            "input_graph", "data_location", "input_height", "input_width",
            "batch_size", "input_layer", "output_layer",
            "num_inter_threads", "num_intra_threads"]

        cmd = self.add_numactl_and_args(accuracy_script, script_args_list)
        os.system(cmd)

    def add_numactl_and_args(self, script_path, script_args_list):
        """
        Helper function for creating string that includes the python script
        and all of the specified args.
        """
        cmd_prefix = "python " + script_path

        if self.args.single_socket:
            cmd_prefix = 'numactl --cpunodebind=0 --membind=0 ' + cmd_prefix

        for arg in vars(self.args):
            arg_value = getattr(self.args, arg)
            if arg == "batch_size" and arg_value == -1:
                continue
            if arg_value and (arg in script_args_list):
                cmd_prefix = '{} --{}={}'.format(cmd_prefix, arg, arg_value)

        print("Executing command: {}".format(cmd_prefix))
        return cmd_prefix

    def run(self):
        # Parse custom arguments and append to self.args
        self.parse_args()
        if self.args.benchmark_only:
            self.run_benchmark()
        if self.args.accuracy_only:
            self.run_accuracy()
