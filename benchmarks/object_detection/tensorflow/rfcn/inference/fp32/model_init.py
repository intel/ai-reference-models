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


class ModelInitializer(BaseModelInitializer):
    args = None
    command_prefix = ""

    def run_inference_sanity_checks(self, args, custom_args):

        if args.batch_size != -1 and args.batch_size != 1:
            sys.exit("R-FCN inference supports 'batch-size=1' " +
                     "only, please modify via the '--batch_size' flag.")

    def __init__(self, args, custom_args, platform_util):
        self.args = args
        self.custom_args = custom_args
        platform_args = platform_util

        self.parse_custom_args()
        if args.mode == "inference":
            self.run_inference_sanity_checks(self.args, self.custom_args)
            benchmark_script = os.path.join(
                self.args.intelai_models, self.args.mode,
                self.args.platform, "eval.py")
            self.command_prefix = "python " + benchmark_script
            if args.single_socket:
                self.command_prefix = "numactl --cpunodebind=" + \
                                      str(self.args.socket_id) + \
                                      " --membind=0 " + self.command_prefix
                args.num_inter_threads = 1
                args.num_intra_threads = platform_args.num_cores_per_socket()
            else:
                args.num_inter_threads = platform_args.num_cpu_sockets()
                if args.num_cores == -1:
                    args.num_intra_threads = \
                        platform_args.num_cores_per_socket() * \
                        args.num_inter_threads
                if self.args.num_cores == 1:
                    self.command_prefix = "taskset -c 0 " + \
                                          self.command_prefix
                    args.num_intra_threads = 1
                else:
                    self.command_prefix = "taskset -c 0-" + \
                                          str(self.args.num_cores - 1) + \
                                          " " + self.command_prefix
                    args.num_intra_threads = args.num_cores

            os.environ["OMP_NUM_THREADS"] = str(args.num_intra_threads)
            self.research_dir = os.path.join(args.model_source_dir, "research")
            config_file_path = os.path.join(self.args.checkpoint,
                                            self.args.config_file)
            self.run_cmd = \
                self.command_prefix + \
                " --inter_op " + str(args.num_inter_threads) + \
                " --intra_op " + str(args.num_intra_threads) + \
                " --omp " + str(args.num_intra_threads) + \
                " --pipeline_config_path " + config_file_path + \
                " --checkpoint_dir " + str(args.checkpoint) + \
                " --eval_dir " + self.research_dir + \
                "/object_detection/models/rfcn/eval " + \
                " --logtostderr " + \
                " --blocktime=0 " + \
                " --run_once=True"
        else:
            # TODO: Add training commands
            sys.exit("Training is currently not supported.")

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
