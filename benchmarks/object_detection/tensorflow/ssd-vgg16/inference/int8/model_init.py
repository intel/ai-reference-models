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

import os
import sys

from common.base_model_init import BaseModelInitializer

os.environ["KMP_BLOCKTIME"] = "0"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"


class ModelInitializer(BaseModelInitializer):
    args = None
    custom_args = []

    def run_inference_sanity_checks(self, args, custom_args):
        if not args.input_graph:
            sys.exit("Please provide a path to the frozen graph directory"
                     " via the '--in-graph' flag.")

    def __init__(self, args, custom_args, platform_util=None):
        self.args = args
        self.custom_args = custom_args
        self.platform_util = platform_util
        self.benchmark_command = ""
        self.run_inference_sanity_checks(self.args, self.custom_args)

        if self.args.single_socket:
            self.args.num_inter_threads = 1
            self.args.num_intra_threads = \
                self.platform_util.num_cores_per_socket() \
                if self.args.num_cores == -1 else self.args.num_cores
            self.benchmark_command = "numactl --cpunodebind=" + \
                                     str(self.args.socket_id) + \
                                     " --membind=" + \
                                     str(self.args.socket_id)
        else:
            self.args.num_inter_threads = \
                self.platform_util.num_cpu_sockets()

            if self.args.num_cores == -1:
                self.args.num_intra_threads = \
                    int(self.platform_util.num_cores_per_socket() *
                        self.platform_util.num_cpu_sockets())
            else:
                self.args.num_intra_threads = self.args.num_cores

        benchmark_script = os.path.join(
            self.args.intelai_models, "ssd-benchmark.py")

        self.benchmark_command = self.benchmark_command + \
            " python " + benchmark_script
        self.benchmark_command = self.benchmark_command + " --graph " + \
            self.args.input_graph + " --num_intra_threads=" + \
            str(self.args.num_intra_threads) + " --num_inter_threads=" + \
            str(self.args.num_inter_threads) + " --batch_size=" + \
            str(self.args.batch_size)

        if self.args.data_location:
            self.benchmark_command = self.benchmark_command + \
                                 " --use_voc_data" + \
                                 " --data_location=" + \
                                 self.args.data_location
        if self.args.accuracy_only:
            self.benchmark_command = \
                self.benchmark_command + " --accuracy_check"

    def run(self):
        self.run_command(self.benchmark_command)
