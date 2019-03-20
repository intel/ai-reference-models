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

import argparse
import os


class ModelInitializer(BaseModelInitializer):
    """Model initializer for resnet50 int8 inference"""

    def __init__(self, args, custom_args=[], platform_util=None):
        super(ModelInitializer, self).__init__(args, custom_args, platform_util)

        # Set the num_inter_threads and num_intra_threads
        self.set_num_inter_intra_threads()
        # Set env vars, if they haven't already been set
        set_env_var("OMP_NUM_THREADS", self.args.num_intra_threads, overwrite_existing=True)

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--warmup-steps", dest="warmup_steps",
            help="number of warmup steps",
            type=int, default=10)
        parser.add_argument(
            "--steps", dest="steps",
            help="number of steps",
            type=int, default=50)
        parser.add_argument(
            '--kmp-blocktime', dest='kmp_blocktime',
            help='number of kmp block time',
            type=int, default=1)
        parser.add_argument(
            "--calibration-only",
            help="Calibrate the accuracy.",
            dest="calibration_only", action="store_true")
        parser.add_argument(
            "--calibrate", dest="calibrate",
            help=" run accuracy with calibration data, "
                 "to generate min_max ranges, calibrate=[True/False]",
            type=bool, default=False)

        self.args = parser.parse_args(self.custom_args,
                                      namespace=self.args)
        # Use default KMP variable values, but override the default KMP_BLOCKTIME value
        self.set_kmp_vars(kmp_blocktime=str(self.args.kmp_blocktime))

    def run_benchmark(self):
        benchmark_script = os.path.join(self.args.intelai_models,
                                        self.args.precision, "benchmark.py")
        script_args_list = [
            "input_graph", "batch_size",
            # comment it out for now since start.sh added data-location=/dataset
            # "data_location",
            "num_inter_threads", "num_intra_threads",
            "data_num_inter_threads", "data_num_intra_threads",
            "warmup_steps", "steps"]

        cmd_prefix = self.get_numactl_command(self.args.socket_id) + \
            "python " + benchmark_script
        cmd = self.add_args_to_command(cmd_prefix, script_args_list)
        # add num_cores
        num_cores = self.platform_util.num_cores_per_socket if self.args.num_cores == -1 \
            else self.args.num_cores
        cmd += " --num_cores=" + str(num_cores)
        # workaround the --data-location problem
        if self.args.data_location and os.listdir(self.args.data_location):
            cmd += " --data_location=" + self.args.data_location
        self.run_command(cmd)

    def run_accuracy(self):
        accuracy_script = os.path.join(self.args.intelai_models,
                                       self.args.precision, "accuracy.py")
        script_args_list = [
            "input_graph", "data_location",
            "batch_size",
            "num_inter_threads", "num_intra_threads", "calibrate"]

        cmd_prefix = self.get_numactl_command(self.args.socket_id) + \
            "python " + accuracy_script
        cmd = self.add_args_to_command(cmd_prefix, script_args_list)
        self.run_command(cmd)

    def run_calibration(self):
        calibration_script = os.path.join(self.args.intelai_models,
                                          self.args.precision,
                                          "generate_calibration_data.py")
        script_args_list = [
            "input_graph", "data_location",
            "batch_size",
            "num_inter_threads", "num_intra_threads"]
        cmd_prefix = self.get_numactl_command(self.args.socket_id) + \
            "python " + calibration_script
        cmd = self.add_args_to_command(cmd_prefix, script_args_list)
        self.run_command(cmd)

    def run(self):
        # Parse custom arguments and append to self.args
        self.parse_args()
        if self.args.benchmark_only:
            self.run_benchmark()
        if self.args.accuracy_only:
            if not self.args.calibration_only:
                self.run_accuracy()
            else:
                self.run_calibration()
