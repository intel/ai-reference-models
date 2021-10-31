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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import argparse
from common.base_model_init import BaseModelInitializer


class ModelInitializer(BaseModelInitializer):
    """Model initializer for Wide and deep large dataset FP32 inference"""

    def __init__(self, args, custom_args=[], platform_util=None):
        super(ModelInitializer, self).__init__(args, custom_args, platform_util)

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--num_omp_threads", dest='num_omp_threads',
                            type=str, default=None,
                            help="number of omp threads")
        parser.add_argument("--use_parallel_batches", dest='use_parallel_batches',
                            type=str, default="False",
                            help="Enable to batches in parallel")
        parser.add_argument("--num_parallel_batches", dest='num_parallel_batches', default="1",
                            type=str, help="num of parallel batches.Default is 1")
        parser.add_argument('--kmp_block_time', dest='kmp_block_time',
                            help='number of kmp block time.',
                            type=str, default=None)
        parser.add_argument('--kmp_settings', dest='kmp_settings',
                            help='kmp settings',
                            type=str, default=None)
        self.args = parser.parse_args(self.custom_args,
                                      namespace=self.args)
        config_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        self.set_kmp_vars(config_file_path, kmp_settings=str(self.args.kmp_settings),
                          kmp_blocktime=str(self.args.kmp_block_time))

    def run_benchmark(self):
        enable_parallel_batches = getattr(self.args, 'use_parallel_batches')
        script_args_list = ["input_graph", "batch_size",
                            "num_inter_threads", "num_intra_threads",
                            "accuracy_only", "data_location", "num_omp_threads"]
        if enable_parallel_batches == 'True':
            benchmark_script = os.path.join(self.args.intelai_models, self.args.mode, "parallel_inference.py")
            script_args_list.append("num_parallel_batches")
        else:
            benchmark_script = os.path.join(self.args.intelai_models, self.args.mode, "inference.py")
        command_prefix = self.get_command_prefix(-1)
        cmd_prefix = command_prefix + self.python_exe + " " + benchmark_script
        cmd = self.add_args_to_command(cmd_prefix, script_args_list)
        self.run_command(cmd)

    def run(self):
        # Parse custom arguments and append to self.args
        self.parse_args()
        self.run_benchmark()
