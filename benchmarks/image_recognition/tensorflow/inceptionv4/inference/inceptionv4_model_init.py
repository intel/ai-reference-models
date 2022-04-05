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

#


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

from common.base_model_init import BaseModelInitializer, set_env_var


class InceptionV4ModelInitializer(BaseModelInitializer):
    """Common model initializer for InceptionV4 inference"""

    def __init__(self, args, custom_args=[], platform_util=None):
        super(InceptionV4ModelInitializer, self).__init__(args, custom_args, platform_util)

        # Environment variables
        set_env_var("OMP_NUM_THREADS", platform_util.num_cores_per_socket
                    if self.args.num_cores == -1 else self.args.num_cores)

        # Set KMP env vars, if they haven't already been set
        config_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        self.set_kmp_vars(config_file_path)

        self.set_num_inter_intra_threads(num_inter_threads=platform_util.num_threads_per_core,
                                         num_intra_threads=platform_util.num_cores_per_socket)

    def parse_args(self):
        if self.custom_args:
            parser = argparse.ArgumentParser()
            parser.add_argument(
                "--input-height", default=None, dest="input_height",
                type=int, help="input height")
            parser.add_argument(
                "--input-width", default=None,
                dest="input_width", type=int, help="input width")
            parser.add_argument(
                "--warmup-steps", dest="warmup_steps",
                help="number of warmup steps", type=int, default=10)
            parser.add_argument(
                "--steps", dest="steps", help="number of steps", type=int,
                default=50)
            parser.add_argument(
                "--input-layer", dest="input_layer",
                help="name of input layer", type=str, default=None)
            parser.add_argument(
                "--output-layer", dest="output_layer",
                help="name of output layer", type=str, default=None)

            parser.add_argument(
                '--kmp-blocktime', dest='kmp_blocktime',
                help='number of kmp block time',
                type=int, default=1)

            self.args = parser.parse_args(self.custom_args, namespace=self.args)

            # Set KMP env vars, if they haven't already been set, but override the default KMP_BLOCKTIME value
            config_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
            self.set_kmp_vars(config_file_path, kmp_blocktime=str(self.args.kmp_blocktime))

    def add_command_prefix(self, script_path):
        """ Uses the specified script path and adds on the command prefix """
        return self.get_command_prefix(self.args.socket_id) + self.python_exe + " " + \
            script_path

    def run_benchmark(self):
        """ Setup the command string and run the benchmarking script """
        benchmark_script = os.path.join(
            self.args.intelai_models, self.args.mode, "benchmark.py")
        script_args_list = [
            "input_graph", "input_height", "input_width", "batch_size",
            "input_layer", "output_layer", "num_inter_threads",
            "num_intra_threads",
            "warmup_steps", "steps"]
        cmd_prefix = self.add_command_prefix(benchmark_script)
        cmd = self.add_args_to_command(cmd_prefix, script_args_list)

        self.run_command(cmd)

    def run_accuracy(self):
        """ Setup the command string and run the accuracy test """
        accuracy_script = os.path.join(
            self.args.intelai_models, self.args.mode, "accuracy.py")
        script_args_list = [
            "input_graph", "data_location", "input_height", "input_width",
            "batch_size", "input_layer", "output_layer",
            "num_inter_threads", "num_intra_threads"]
        cmd_prefix = self.add_command_prefix(accuracy_script)
        cmd = self.add_args_to_command(cmd_prefix, script_args_list)

        self.run_command(cmd)

    def run(self):
        self.parse_args()
        if self.args.benchmark_only:
            self.run_benchmark()
        if self.args.accuracy_only:
            self.run_accuracy()
