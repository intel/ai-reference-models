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
    """initialize mode and run benchmark"""

    def __init__(self, args, custom_args=[], platform_util=None):
        super(ModelInitializer, self).__init__(args, custom_args, platform_util)

        arg_parser = argparse.ArgumentParser(description='Parse additional args')
        arg_parser.add_argument("--dataset", type=str, default="ml-1m",
                                choices=["ml-1m", "ml-20m"],
                                help=("Dataset to be trained and evaluated."),
                                dest="dataset")
        self.additional_args, unknown_args = arg_parser.parse_known_args(custom_args)

        self.benchmark_command = ""

        # Add custom parameters for specific dataset
        if self.additional_args.dataset == "ml-20m":
            self.additional_args.dataset = " --dataset=ml-20m " + \
                " --layers=256,256,128,64 --num_factors=64" + \
                " --eval_batch_size 160000" + \
                " --learning_rate 0.003821" + \
                " --beta1 0.783529 --beta2 0.909003 --epsilon 1.45439e-07" + \
                " --hr_threshold 0.635" + \
                " --ml_perf"

        # use default batch size if -1
        if self.args.batch_size == -1:
            self.args.batch_size = 98304

        # Set KMP env vars, if they haven't already been set
        config_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        self.set_kmp_vars(config_file_path)

        # set num_inter_threads and num_intra_threads
        self.set_num_inter_intra_threads()

        benchmark_script = os.path.join(
            self.args.intelai_models, self.args.mode, "ncf_estimator_main.py")

        self.benchmark_command = self.get_command_prefix(args.socket_id) + \
            self.python_exe + " " + benchmark_script

        set_env_var("OMP_NUM_THREADS", self.args.num_intra_threads)
        set_env_var("TF_NUM_INTRAOP_THREADS", self.args.num_intra_threads)
        set_env_var("TF_NUM_INTEROP_THREADS", self.args.num_inter_threads)

        self.benchmark_command = self.benchmark_command + \
            " -dd=" + str(args.data_location) + \
            " -md=" + str(args.checkpoint) + \
            " -bs=" + str(self.args.batch_size) + \
            " -hk=examplespersecondhook" + \
            self.additional_args.dataset + ' ' + ' '.join(unknown_args)

    def run(self):
        if self.benchmark_command:
            self.run_command(self.benchmark_command)
