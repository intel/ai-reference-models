#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019-2021 Intel Corporation
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

import argparse
import os


class ModelInitializer(BaseModelInitializer):
    """Model initializer for resnet50 int8 inference"""

    def __init__(self, args, custom_args=[], platform_util=None):
        super(ModelInitializer, self).__init__(args, custom_args, platform_util)

        # Set the num_inter_threads and num_intra_threads
        self.set_num_inter_intra_threads()

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

        # Set KMP env vars, if they haven't already been set, but override the default KMP_BLOCKTIME value
        config_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        self.set_kmp_vars(config_file_path, kmp_blocktime=str(self.args.kmp_blocktime))

        if not self.args.gpu:
            set_env_var("OMP_NUM_THREADS", self.args.num_intra_threads)

    def run_benchmark_or_accuracy(self):
        # If weight-sharing flag is ON, then use the weight-sharing script.
        if self.args.weight_sharing and not self.args.accuracy_only:
            cmd = os.path.join(
                self.args.intelai_models, self.args.mode, "cpu",
                "eval_image_classifier_inference_weight_sharing.py")
        else:
            if self.args.gpu:
                cmd = os.path.join(
                    self.args.intelai_models, self.args.mode, "gpu", self.args.precision,
                    "eval_image_classifier_inference.py")
            else:
                cmd = os.path.join(
                    self.args.intelai_models, self.args.mode, "cpu",
                    "eval_image_classifier_inference.py")

        cmd = self.get_command_prefix(self.args.socket_id) + self.python_exe + " " + cmd

        if self.args.gpu:
            cmd += " --input-graph=" + self.args.input_graph + \
                   " --batch-size=" + str(self.args.batch_size) + \
                   " --warmup-steps=" + str(self.args.warmup_steps) + \
                   " --steps=" + str(self.args.steps)
        else:
            cmd += " --input-graph=" + self.args.input_graph + \
                   " --num-inter-threads=" + str(self.args.num_inter_threads) + \
                   " --num-intra-threads=" + str(self.args.num_intra_threads) + \
                   " --batch-size=" + str(self.args.batch_size) + \
                   " --warmup-steps=" + str(self.args.warmup_steps) + \
                   " --steps=" + str(self.args.steps)

        if self.args.calibrate:
            cmd += " --calibrate=" + str(self.args.calibrate)
        if self.args.data_num_inter_threads:
            cmd += " --data-num-inter-threads=" + str(self.args.data_num_inter_threads)
        if self.args.data_num_intra_threads:
            cmd += " --data-num-intra-threads=" + str(self.args.data_num_intra_threads)

        # if the data location directory is not empty, then include the arg
        if self.args.data_location and os.listdir(self.args.data_location):
            cmd += " --data-location=" + self.args.data_location

        # enable onednn graph
        if self.args.onednn_graph:
            cmd += " --onednn-graph"

        if self.args.accuracy_only:
            cmd += " --accuracy-only"
        self.run_command(cmd)

    def run_calibration(self):
        calibration_script = os.path.join(self.args.intelai_models,
                                          self.args.precision,
                                          "generate_calibration_data.py")
        script_args_list = [
            "input_graph", "data_location",
            "batch_size",
            "num_inter_threads", "num_intra_threads"]
        cmd_prefix = self.get_command_prefix(self.args.socket_id) + \
            self.python_exe + " " + calibration_script
        cmd = self.add_args_to_command(cmd_prefix, script_args_list)
        self.run_command(cmd)

    def run(self):
        # Parse custom arguments and append to self.args
        self.parse_args()
        if self.args.accuracy_only and self.args.calibration_only:
            self.run_calibration()
        else:
            self.run_benchmark_or_accuracy()
