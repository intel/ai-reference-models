#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Intel Corporation
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


class ModelInitializer(BaseModelInitializer):
    """initialize model and run benchmark"""

    def __init__(self, args, custom_args=[], platform_util=None):
        self.args = args
        self.custom_args = custom_args
        self.platform_util = platform_util

        # set num_inter_threads and num_intra_threads
        self.set_default_inter_intra_threads(self.platform_util)

        # Set KMP env vars, if they haven't already been set
        self.set_kmp_vars()
        set_env_var("KMP_HW_SUBSET", "1T")

        benchmark_script = os.path.join(
            self.args.intelai_models, args.mode, args.precision,
            "inference_bench.py")
        self.benchmark_command = self.get_numactl_command(args.socket_id) + \
            "python " + benchmark_script

        set_env_var("OMP_NUM_THREADS", self.args.num_intra_threads)
        self.cifar10_dir = os.path.join(args.model_source_dir,
                                        "research", "gan", "cifar")

        self.benchmark_command = self.benchmark_command + \
            " -ckpt " + str(self.args.checkpoint) + \
            " -dl " + str(self.args.data_location) + \
            " --num_inter_threads " + str(self.args.num_inter_threads) + \
            " --num_intra_threads " + str(self.args.num_intra_threads) + \
            " -nw 100 -nb 500" + \
            " --bs " + str(self.args.batch_size) + \
            " --kmp_blocktime " + os.environ["KMP_BLOCKTIME"] + \
            " --kmp_settings " + os.environ["KMP_SETTINGS"]

    def run(self):
        if self.benchmark_command:
            original_dir = os.getcwd()
            os.chdir(self.cifar10_dir)
            self.run_command(self.benchmark_command)
            os.chdir(original_dir)
