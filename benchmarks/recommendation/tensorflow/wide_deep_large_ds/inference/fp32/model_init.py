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
    """Model initializer for Wide and deep large dataset FP32 inference"""

    def __init__(self, args, custom_args=[], platform_util=None):
        super(ModelInitializer, self).__init__(args, custom_args, platform_util)
        # Set the num_inter_threads and num_intra_threads
        self.set_num_inter_intra_threads(num_inter_threads=platform_util.num_cores_per_socket,
                                         num_intra_threads=1)
        # Use default KMP AFFINITY values, override KMP_BLOCKTIME & enable KMP SETTINGS
        self.set_kmp_vars(kmp_settings="1", kmp_blocktime="0",
                          kmp_affinity="noverbose,warnings,respect,granularity=core,none")

        # Set env vars, if they haven't already been set
        set_env_var("OMP_NUM_THREADS", self.args.num_intra_threads)

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--num-parallel-batches", default=-1,
            type=int, help="num of parallel batches")

        self.args = parser.parse_args(self.custom_args,
                                      namespace=self.args)

    def run_benchmark(self):
        benchmark_script = os.path.join(self.args.intelai_models,
                                        self.args.mode, "inference.py")
        if self.args.num_parallel_batches == -1:
            self.args.num_parallel_batches = self.platform_util.num_cores_per_socket

        script_args_list = ["input_graph", "num_parallel_batches", "batch_size",
                            "num_inter_threads", "num_intra_threads", "accuracy_only", "data_location"]

        cmd_prefix = self.get_numactl_command(self.args.socket_id) + \
            "python " + benchmark_script
        cmd = self.add_args_to_command(cmd_prefix, script_args_list)
        self.run_command(cmd)

    def run(self):
        # Parse custom arguments and append to self.args
        self.parse_args()
        self.run_benchmark()
