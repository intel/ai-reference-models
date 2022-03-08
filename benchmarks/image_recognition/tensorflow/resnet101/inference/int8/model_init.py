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
from common.base_model_init import BaseModelInitializer
from common.base_model_init import set_env_var

import os
import argparse


class ModelInitializer(BaseModelInitializer):
    """Model initializer for resnet101 int8 inference"""

    def __init__(self, args, custom_args=[], platform_util=None):
        super(ModelInitializer, self).__init__(args, custom_args, platform_util)

        # Parse custom arguments and append to self.args
        self.parse_args()

        # Set the num_inter_threads and num_intra_threads
        self.set_num_inter_intra_threads()

        # Set env vars, if they haven't already been set
        set_env_var("OMP_NUM_THREADS",
                    platform_util.num_cores_per_socket if args.num_cores == -1 else args.num_cores)

        # Set KMP env vars, if they haven't already been set
        config_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        self.set_kmp_vars(config_file_path)
        self.set_kmp_vars(config_file_path, kmp_blocktime=str(self.args.kmp_blocktime))

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--input-height", default=None,
                            dest='input_height', type=int,
                            help="input height")
        parser.add_argument("--input-width", default=None,
                            dest='input_width', type=int,
                            help="input width")
        parser.add_argument('--warmup-steps', dest='warmup_steps',
                            help='number of warmup steps', type=int,
                            default=40)
        parser.add_argument('--steps', dest='steps',
                            help='number of steps', type=int,
                            default=100)
        parser.add_argument('--input-layer', dest='input_layer',
                            help='name of input layer', type=str,
                            default=None)
        parser.add_argument('--output-layer', dest='output_layer',
                            help='name of output layer', type=str,
                            default=None)
        parser.add_argument(
            "--calibration-only",
            help="Calibrate the accuracy.",
            dest="calibration_only", action="store_true")
        parser.add_argument(
            '--kmp-blocktime', dest='kmp_blocktime',
            help='number of kmp block time',
            type=int, default=1)

        self.args = parser.parse_args(self.custom_args,
                                      namespace=self.args)

    def run_benchmark_or_accuracy(self):
        cmd = os.path.join(
            self.args.intelai_models, self.args.mode,
            "eval_image_classifier_inference.py")

        cmd = self.get_command_prefix(self.args.socket_id) + self.python_exe + " " + cmd

        cmd += " --input-graph=" + self.args.input_graph + \
               " --num-inter-threads=" + str(self.args.num_inter_threads) + \
               " --num-intra-threads=" + str(self.args.num_intra_threads) + \
               " --batch-size=" + str(self.args.batch_size) + \
               " --warmup-steps=" + str(self.args.warmup_steps) + \
               " --steps=" + str(self.args.steps)

        if self.args.data_num_inter_threads:
            cmd += " --data-num-inter-threads=" + str(self.args.data_num_inter_threads)
        if self.args.data_num_intra_threads:
            cmd += " --data-num-intra-threads=" + str(self.args.data_num_intra_threads)

        # if the data location directory is not empty, then include the arg
        if self.args.data_location and os.listdir(self.args.data_location):
            cmd += " --data-location=" + self.args.data_location
        if self.args.accuracy_only:
            cmd += " --accuracy-only"

        self.run_command(cmd)

    def run_calibration(self):
        calibration_script = os.path.join(self.args.intelai_models, self.args.mode,
                                          self.args.precision, "calibration.py")
        script_args_list = [
            "input_graph", "data_location",
            "batch_size",
            "num_inter_threads", "num_intra_threads"]
        cmd_prefix = self.get_command_prefix(self.args.socket_id) + \
            self.python_exe + " " + calibration_script
        cmd = self.add_args_to_command(cmd_prefix, script_args_list)
        self.run_command(cmd)

    def run(self):

        if self.args.accuracy_only and self.args.calibration_only:
            self.run_calibration()
        else:
            self.run_benchmark_or_accuracy()
