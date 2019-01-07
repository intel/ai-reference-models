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

os.environ["KMP_BLOCKTIME"] = "1"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"


class ModelInitializer (BaseModelInitializer):
    def run_inference_sanity_checks(self, args, custom_args):
        if args.batch_size != -1 and args.batch_size != 1:
            sys.exit("Fast R-CNN inference supports 'batch-size=1' " +
                     "only, please modify via the '--batch_size' flag.")

    def __init__(self, args, custom_args, platform_util=None):
        self.args = args
        self.custom_args = custom_args
        self.platform_util = platform_util

        self.parse_custom_args()
        self.run_inference_sanity_checks(self.args, self.custom_args)

        benchmark_script = os.path.join(
            self.args.intelai_models, self.args.mode, self.args.precision,
            "eval.py")
        self.command_prefix = self.get_numactl_command(self.args.socket_id) + \
            "python " + benchmark_script
        if self.args.socket_id != -1:
            self.args.num_inter_threads = 1
            self.args.num_intra_threads = \
                self.platform_util.num_cores_per_socket()

        os.environ["OMP_NUM_THREADS"] = str(self.args.num_intra_threads)
        self.research_dir = os.path.join(self.args.model_source_dir,
                                         "research")
        config_file_path = os.path.join(self.args.checkpoint,
                                        self.args.config_file)
        self.run_cmd = \
            self.command_prefix + \
            " --num_inter_threads " + str(self.args.num_inter_threads) + \
            " --num_intra_threads " + str(self.args.num_intra_threads) + \
            " --pipeline_config_path " + config_file_path + \
            " --checkpoint_dir " + str(args.checkpoint) + \
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
