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

from common.base_model_init import BaseModelInitializer


class ModelInitializer(BaseModelInitializer):
    """initialize mode and run benchmark"""

    def __init__(self, args, custom_args=[], platform_util=None):
        self.args = args
        self.custom_args = custom_args
        if not platform_util:
            raise ValueError("Did not find any platform info.")

        # Environment variables
        if self.args.num_cores == -1:
            os.environ["OMP_NUM_THREADS"] = \
                str(platform_util.num_cores_per_socket())
        else:
            os.environ["OMP_NUM_THREADS"] = str(self.args.num_cores)
        os.environ["KMP_BLOCKTIME"] = "0"
        os.environ["KMP_SETTINGS"] = "1"
        os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"

        self.parse_args()
        if self.args.benchmark_only:
            run_script = os.path.join(
                args.intelai_models, args.precision, "benchmark.py")
            script_args_list = [
                "input_graph", "input_height", "input_width", "batch_size",
                "input_layer", "output_layer", "num_inter_threads",
                "num_intra_threads", "warmup_steps", "steps"]
        if self.args.accuracy_only:
            run_script = os.path.join(
                args.intelai_models, args.precision, "accuracy.py")
            script_args_list = [
                "input_graph", "data_location", "input_height", "input_width",
                "batch_size", "input_layer", "output_layer",
                "num_inter_threads", "num_intra_threads"]

        self.cmd = self.get_numactl_command(args.socket_id) + \
            "python " + run_script

        self.cmd = self.add_args_to_command(self.cmd, script_args_list)

    def parse_args(self):
        if self.custom_args:
            parser = argparse.ArgumentParser()
            parser.add_argument(
                "--input-height", default=None,
                dest="input_height", type=int, help="input height")
            parser.add_argument(
                "--input-width", default=None,
                dest="input_width", type=int, help="input width")
            parser.add_argument(
                "--warmup-steps", dest="warmup_steps",
                help="number of warmup steps", type=int, default=10)
            parser.add_argument(
                "--steps", dest="steps",
                help="number of steps", type=int, default=200)
            parser.add_argument(
                "--num-inter-threads", dest="num_inter_threads",
                help="number threads across operators", type=int, default=1)
            parser.add_argument(
                "--num-intra-threads", dest="num_intra_threads",
                help="number threads for an operator", type=int, default=1)
            parser.add_argument(
                "--input-layer", dest="input_layer",
                help="name of input layer", type=str, default=None)
            parser.add_argument(
                "--output-layer", dest="output_layer",
                help="name of output layer", type=str, default=None)

            self.args = parser.parse_args(self.custom_args,
                                          namespace=self.args)

    def run(self):
        self.run_command(self.cmd)
