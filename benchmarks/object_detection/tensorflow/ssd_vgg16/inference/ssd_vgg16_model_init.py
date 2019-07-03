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
# SPDX-License-Identifier: EPL-2.0
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse

from common.base_model_init import BaseModelInitializer, set_env_var


class SSDVGG16ModelInitializer(BaseModelInitializer):
    """Common model initializer for SSD-VGG16 inference"""

    def run_inference_sanity_checks(self, args, custom_args):
        if not args.input_graph:
            sys.exit("Please provide a path to the frozen graph directory"
                     " via the '--in-graph' flag.")
        if not args.data_location and self.args.accuracy_only:
            sys.exit("For accuracy test, please provide a path to the data directory via the "
                     "'--data-location' flag.")
        if args.batch_size != -1 and args.batch_size != 1:
            sys.exit("SSD-VGG16 inference supports 'batch-size=1' " +
                     "only, please modify via the '--batch_size' flag.")

    def __init__(self, args, custom_args, platform_util):
        super(SSDVGG16ModelInitializer, self).__init__(args, custom_args, platform_util)

        self.parse_custom_args()
        self.run_inference_sanity_checks(self.args, self.custom_args)

        # Set KMP env vars, if they haven't already been set
        config_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        self.set_kmp_vars(config_file_path)

        self.set_num_inter_intra_threads(num_inter_threads=self.args.num_inter_threads,
                                         num_intra_threads=self.args.num_intra_threads)

        omp_num_threads = str(int(platform_util.num_cores_per_socket / 2))\
            if self.args.precision == "int8" else platform_util.num_cores_per_socket

        set_env_var("OMP_NUM_THREADS", omp_num_threads
                    if self.args.num_cores == -1 else self.args.num_cores)

        script_path = os.path.join(
            self.args.intelai_models, self.args.mode, "eval_ssd.py")

        self.run_cmd = self.get_command_prefix(
            self.args.socket_id) + "{} {}".format(self.python_exe, script_path)

        self.run_cmd += " --input-graph={} " \
                        " --num-inter-threads={} --num-intra-threads={} ". \
            format(self.args.input_graph, self.args.num_inter_threads,
                   self.args.num_intra_threads)

        if self.args.data_num_inter_threads:
            self.run_cmd += " --data-num-inter-threads={} ".format(
                self.args.data_num_inter_threads)

        if self.args.data_num_intra_threads:
            self.run_cmd += " --data-num-intra-threads={} ".format(
                self.args.data_num_intra_threads)

        if self.args.benchmark_only:
            self.run_cmd += " --warmup-steps={} --steps={} ". \
                format(self.args.warmup_steps, self.args.steps)

        # if the data location directory is not empty, then include the arg
        if self.args.data_location and os.listdir(self.args.data_location):
            self.run_cmd += " --data-location={} ".format(self.args.data_location)

        if self.args.accuracy_only:
            self.run_cmd += "--accuracy-only "

    def parse_custom_args(self):
        if self.custom_args:
            parser = argparse.ArgumentParser()
            parser.add_argument("--warmup-steps", type=int, default=10,
                                help="number of warmup steps")
            parser.add_argument("--steps", type=int, default=50,
                                help="number of steps")

            self.args = parser.parse_args(self.custom_args,
                                          namespace=self.args)

    def run(self):
        self.run_command(self.run_cmd)
