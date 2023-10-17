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

import os
import argparse
from common.base_model_init import BaseModelInitializer, set_env_var


class ModelInitializer(BaseModelInitializer):
    # SSD-MobileNet Int8 inference model initialization
    args = None
    custom_args = []

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
        self.args = parser.parse_args(self.custom_args,
                                      namespace=self.args)

    def __init__(self, args, custom_args=[], platform_util=None):
        super(ModelInitializer, self).__init__(args, custom_args, platform_util)
        self.parse_args()
        # Set the num_inter_threads and num_intra_threads
        # if user did not provide then default value based on platform will be set
        self.set_num_inter_intra_threads(self.args.num_inter_threads,
                                         self.args.num_intra_threads)
        # Set KMP env vars, if they haven't already been set
        config_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        self.set_kmp_vars(config_file_path)
        if self.args.gpu:
            benchmark_script = os.path.join(self.args.intelai_models, self.args.mode, "gpu",
                                            self.args.precision, "infer_detections.py")
        else:
            benchmark_script = os.path.join(self.args.intelai_models, self.args.mode,
                                            self.args.precision, "infer_detections.py")
        self.command_prefix = self.get_command_prefix(self.args.socket_id) \
            + "{} {}".format(self.python_exe, benchmark_script)
        set_env_var("OMP_NUM_THREADS", self.args.num_intra_threads)

        self.command_prefix += " -g {0}".format(self.args.input_graph)
        self.command_prefix += " -i {0}".format(self.args.steps)
        self.command_prefix += " -w {0}".format(self.args.warmup_steps)
        self.command_prefix += " -a {0}".format(self.args.num_intra_threads)
        self.command_prefix += " -e {0}".format(self.args.num_inter_threads)
        if self.args.data_location:
            self.command_prefix += " -d {0}".format(self.args.data_location)

        if self.args.onednn_graph:
            self.command_prefix += " --onednn-graph"

        if self.args.accuracy_only:
            self.command_prefix += " -r"
            assert self.args.data_location, "accuracy must provide the data."
        else:
            # Did not support multi-batch accuracy check.
            self.command_prefix += " -b {0}".format(self.args.batch_size)
            if self.args.gpu:
                self.command_prefix += " --benchmark"

    def run(self):
        self.run_command(self.command_prefix)
