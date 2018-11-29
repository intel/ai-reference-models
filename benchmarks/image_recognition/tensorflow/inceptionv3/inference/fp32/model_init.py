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

import os

os.environ["KMP_BLOCKTIME"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
os.environ["KMP_SETTINGS"] = "1"


class ModelInitializer(BaseModelInitializer):
    """initialize mode and run benchmark"""

    def __init__(self, args, custom_args=[], platform_util=None):

        self.args = args
        self.custom_args = custom_args
        self.platform_util = platform_util
        self.benchmark_command = ""
        if not platform_util:
            raise ValueError("Did not find any platform info.")

        # use default batch size if -1
        if self.args.batch_size == -1:
            self.args.batch_size = 128

        self.args.num_inter_threads = 1
        self.args.num_intra_threads = \
            self.platform_util.num_cores_per_socket()

        if not self.args.single_socket:
            self.args.num_intra_threads *= \
                self.platform_util.num_cpu_sockets()
            self.args.num_inter_threads = 2

        if self.args.mode == "inference":
            benchmark_script = os.path.join(
                self.args.intelai_models, self.args.platform,
                "eval_image_classifier_inference.py")

            self.benchmark_command = "python " + benchmark_script

            if self.args.single_socket:
                socket_id_str = str(self.args.socket_id)
                self.benchmark_command = \
                    "numactl --cpunodebind=" + socket_id_str + \
                    " --membind=" + socket_id_str + " " + \
                    self.benchmark_command

            os.environ["OMP_NUM_THREADS"] = \
                str(self.args.num_intra_threads)

            self.benchmark_command = \
                self.benchmark_command + \
                " --input-graph=" + self.args.input_graph + \
                " --model-name=" + str(self.args.model_name) + \
                " --inter-op-parallelism-threads=" + \
                str(self.args.num_inter_threads) + \
                " --intra-op-parallelism-threads=" + \
                str(self.args.num_intra_threads) + \
                " --batch-size=" + str(self.args.batch_size)
            if (self.args.data_location):
                self.benchmark_command += \
                    " --data-location=" + self.args.data_location

        else:
            print("Training is not supported currently.")

    def run(self):
        if self.benchmark_command:
            self.run_command(self.benchmark_command)
