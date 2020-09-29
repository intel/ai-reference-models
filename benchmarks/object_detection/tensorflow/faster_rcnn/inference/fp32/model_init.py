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

import argparse
import os
import sys

from common.base_model_init import BaseModelInitializer
from common.base_model_init import set_env_var


class ModelInitializer (BaseModelInitializer):
    def run_inference_sanity_checks(self, args, custom_args):
        if args.batch_size != -1 and args.batch_size != 1:
            sys.exit("Faster R-CNN inference supports 'batch-size=1' " +
                     "only, please modify via the '--batch_size' flag.")

    def __init__(self, args, custom_args, platform_util=None):
        super(ModelInitializer, self).__init__(args, custom_args, platform_util)

        self.research_dir = os.path.join(self.args.model_source_dir,
                                         "research")
        self.run_inference_sanity_checks(self.args, self.custom_args)

        # set num_inter_threads and num_intra_threads
        self.set_num_inter_intra_threads()

        # Set KMP env vars, if they haven't already been set
        config_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        self.set_kmp_vars(config_file_path)

        set_env_var("OMP_NUM_THREADS", self.args.num_intra_threads)

        if self.args.accuracy_only:
            accuracy_script = os.path.join(
                self.args.intelai_models, self.args.mode, self.args.precision,
                "coco_accuracy.sh")
            if not os.path.exists(accuracy_script):
                raise ValueError("Unable to locate the Faster R-CNN accuracy "
                                 "script: {}".format(accuracy_script))
            self.run_cmd = "bash {} {} {}/coco_val.record {}".format(
                accuracy_script, self.args.input_graph,
                self.args.data_location, self.args.model_source_dir)
        else:
            self.parse_custom_args()

            benchmark_script = os.path.join(
                self.args.intelai_models, self.args.mode, self.args.precision,
                "eval.py")
            self.command_prefix = \
                self.get_command_prefix(self.args.socket_id) + self.python_exe + " " + \
                benchmark_script

            config_file_path = os.path.join(self.args.checkpoint,
                                            self.args.config_file)
            self.run_cmd = \
                self.command_prefix + \
                " --num_inter_threads " + \
                str(self.args.num_inter_threads) + \
                " --num_intra_threads " + \
                str(self.args.num_intra_threads) + \
                " --pipeline_config_path " + \
                config_file_path + \
                " --checkpoint_dir " + \
                str(args.checkpoint) + \
                " --eval_dir " + self.research_dir + \
                "/object_detection/log/eval"

    def parse_custom_args(self):
        if self.custom_args:
            parser = argparse.ArgumentParser()
            parser.add_argument("--config_file", default=None,
                                dest="config_file", type=str)
            self.args = parser.parse_args(self.custom_args,
                                          namespace=self.args)

    def run(self):
        original_dir = os.getcwd()
        os.chdir(self.research_dir)
        self.run_command(self.run_cmd)
        os.chdir(original_dir)
