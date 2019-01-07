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
os.environ["KMP_AFFINITY"] = \
    "granularity=fine,verbose,compact,1,0"


class ModelInitializer(BaseModelInitializer):
    def run_inference_sanity_checks(self, args, custom_args):
        if not args.input_graph:
            sys.exit("Please provide a path to the frozen graph directory"
                     " via the '--in-graph' flag.")
        if args.socket_id == -1 and args.num_cores == -1:
            print("***Warning***: Running inference on all cores could degrade"
                  " performance. Pass a '--socket-id' to specify running on"
                  " a single socket instead.\n")

    def __init__(self, args, custom_args, platform_util):
        self.args = args
        self.custom_args = custom_args

        self.run_inference_sanity_checks(self.args, self.custom_args)

        # set num_inter_threads and num_intra_threads
        self.set_default_inter_intra_threads(platform_util)

        benchmark_script = os.path.join(
            self.args.intelai_models, "ssd-benchmark.py")

        self.benchmark_command = self.get_numactl_command(args.socket_id) + \
            "python " + benchmark_script

        os.environ["OMP_NUM_THREADS"] = str(self.args.num_intra_threads)

        self.benchmark_command = self.benchmark_command + \
            " --graph=" + str(self.args.input_graph) + \
            " --num_intra_threads=" + str(self.args.num_intra_threads) + \
            " --num_inter_threads=" + str(self.args.num_inter_threads) + \
            " --batch_size=" + str(self.args.batch_size)

        if self.args.data_location:
            self.benchmark_command = self.benchmark_command + \
                                     " --use_voc_data" + \
                                     " --data_location=" + \
                                     self.args.data_location
        if self.args.accuracy_only:
            self.benchmark_command = self.benchmark_command + \
                                     " --accuracy_check"

    def run(self):
        self.run_command(self.benchmark_command)
