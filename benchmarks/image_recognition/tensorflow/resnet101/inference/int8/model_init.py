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
from common.base_model_init import BaseModelInitializer
from common.base_model_init import set_env_var

import os
import argparse


class ModelInitializer(BaseModelInitializer):
    """Model initializer for resnet101 int8 inference"""

    def __init__(self, args, custom_args=[], platform_util=None):
        super(ModelInitializer, self).__init__(args, custom_args, platform_util)

        # Set the num_inter_threads and num_intra_threads
        self.set_num_inter_intra_threads()

        # Set env vars, if they haven't already been set
        set_env_var("OMP_NUM_THREADS", platform_util.num_cores_per_socket()
                    if args.num_cores == -1 else args.num_cores)

        # Set KMP env vars, but override default KMP_BLOCKTIME value
        self.set_kmp_vars(kmp_blocktime="0")

    def parse_args(self):
        if self.custom_args:
            parser = argparse.ArgumentParser()
            parser.add_argument("--input-height", default=None,
                                dest='input_height', type=int,
                                help="input height")
            parser.add_argument("--input-width", default=None,
                                dest='input_width', type=int,
                                help="input width")
            parser.add_argument('--warmup-steps', dest='warmup_steps',
                                help='number of warmup steps', type=int,
                                default=10)
            parser.add_argument('--steps', dest='steps',
                                help='number of steps', type=int,
                                default=200)
            parser.add_argument('--input-layer', dest='input_layer',
                                help='name of input layer', type=str,
                                default=None)
            parser.add_argument('--output-layer', dest='output_layer',
                                help='name of output layer', type=str,
                                default=None)

            self.args = parser.parse_args(self.custom_args,
                                          namespace=self.args)

    def run_benchmark(self):
        benchmark_script = os.path.join(self.args.intelai_models,
                                        self.args.precision, "benchmark.py")
        script_args_list = ["input_graph", "input_height", "input_width",
                            "batch_size", "input_layer", "output_layer",
                            "num_inter_threads", "num_intra_threads",
                            "warmup_steps", "steps"]
        cmd_prefix = self.get_numactl_command(self.args.socket_id) +\
            "python " + benchmark_script
        cmd = self.add_args_to_command(cmd_prefix, script_args_list)
        self.run_command(cmd)

    def run_accuracy(self):
        accuracy_script = os.path.join(self.args.intelai_models,
                                       self.args.precision, "accuracy.py")
        script_args_list = ["input_graph", "data_location", "input_height",
                            "input_width", "batch_size", "input_layer",
                            "output_layer", "num_inter_threads",
                            "num_intra_threads"]
        cmd_prefix = self.get_numactl_command(self.args.socket_id) + \
            "python " + accuracy_script

        cmd = self.add_args_to_command(cmd_prefix, script_args_list)
        self.run_command(cmd)

    def run(self):
        # Parse custom arguments and append to self.args
        self.parse_args()
        if self.args.benchmark_only:
            self.run_benchmark()
        if self.args.accuracy_only:
            self.run_accuracy()
