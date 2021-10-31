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

#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

from common.base_model_init import BaseModelInitializer
from common.base_model_init import set_env_var


class ModelInitializer(BaseModelInitializer):
    """Model initializer for Mobilenet INT8 inference"""

    def __init__(self, args, custom_args=[], platform_util=None):
        super(ModelInitializer, self).__init__(args, custom_args, platform_util)
        self.cmd = self.get_command_prefix(self.args.socket_id) + "python "

        # Set KMP env vars, if they haven't already been set
        config_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        self.set_kmp_vars(config_file_path)

        # Set the num_inter_threads and num_intra_threads
        self.set_num_inter_intra_threads()
        # Set env vars, if they haven't already been set
        set_env_var("OMP_NUM_THREADS", self.args.num_intra_threads)

        self.parse_args()

        if self.args.benchmark_only:
            run_script = os.path.join(
                self.args.intelai_models, self.args.mode,
                self.args.precision, "benchmark.py")
            script_args_list = [
                "input_graph", "input_height", "input_width", "batch_size",
                "input_layer", "output_layer", "num_inter_threads",
                "num_intra_threads", "warmup_steps", "steps"]

        if hasattr(self.args, 'calibration_only') and self.args.calibration_only:
            run_script = os.path.join(
                self.args.intelai_models, self.args.mode,
                self.args.precision, "calibration.py")
            script_args_list = [
                "input_graph", "data_location", "input_height", "input_width",
                "batch_size", "input_layer", "output_layer",
                "num_inter_threads", "num_intra_threads"]
        elif self.args.accuracy_only:
            run_script = os.path.join(
                self.args.intelai_models, self.args.mode,
                self.args.precision, "accuracy.py")
            script_args_list = [
                "input_graph", "data_location", "input_height", "input_width",
                "batch_size", "input_layer", "output_layer",
                "num_inter_threads", "num_intra_threads"]

        self.cmd = self.add_args_to_command(self.cmd + run_script, script_args_list)

    def parse_args(self):
        if self.custom_args:
            parser = argparse.ArgumentParser()
            parser.add_argument(
                "--input_height", default=224,
                dest='input_height', type=int, help="input height")
            parser.add_argument(
                "--input_width", default=224,
                dest='input_width', type=int, help="input width")
            parser.add_argument(
                "--warmup_steps", dest="warmup_steps",
                help="number of warmup steps",
                type=int, default=10)
            parser.add_argument(
                "--steps", dest="steps",
                help="number of steps",
                type=int, default=50)
            parser.add_argument(
                "--input_layer", dest="input_layer",
                help="name of input layer",
                type=str, default="input")
            parser.add_argument(
                "--output_layer", dest="output_layer",
                help="name of output layer",
                type=str, default="MobilenetV1/Predictions/Reshape_1")
            parser.add_argument(
                "--calibration-only", dest="calibration_only",
                help="calibrate the accuracy",
                action="store_true")

            self.args = parser.parse_args(self.custom_args,
                                          namespace=self.args)

    def run(self):
        if self.cmd:
            self.run_command(self.cmd)
