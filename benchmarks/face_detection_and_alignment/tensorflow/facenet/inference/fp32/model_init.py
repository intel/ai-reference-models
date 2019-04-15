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

from common.base_model_init import BaseModelInitializer
from common.base_model_init import set_env_var

import os
from argparse import ArgumentParser


class ModelInitializer(BaseModelInitializer):
    """initialize mode and run benchmark for FaceNet model"""

    def __init__(self, args, custom_args=[], platform_util=None):
        super(ModelInitializer, self).__init__(args, custom_args, platform_util)
        self.cmd = self.get_numactl_command(self.args.socket_id) + \
            self.python_exe + " "

        # Set KMP env vars, if they haven't already been set
        config_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        self.set_kmp_vars(config_file_path)

        pairs_file = os.path.join(self.args.model_source_dir,
                                  "data/pairs.txt")
        arg_parser = ArgumentParser(description='Parse custom args')
        arg_parser.add_argument(
            "--lfw_pairs", type=str,
            help="The file containing the pairs to use for validation.",
            dest="lfw_pairs", default=pairs_file)
        self.args = arg_parser.parse_args(
            self.custom_args, namespace=self.args)

        # use default batch size if -1
        if self.args.batch_size == -1 or self.args.accuracy_only:
            self.args.batch_size = 100

        # set num_inter_threads and num_intra_threads
        if self.args.batch_size > 32:
            self.set_num_inter_intra_threads(num_inter_threads=2)
        else:
            self.set_num_inter_intra_threads(num_inter_threads=1)

        set_env_var("OMP_NUM_THREADS", self.args.num_intra_threads)

        run_script = os.path.join(self.args.model_source_dir,
                                  "src/validate_on_lfw.py")

        warmup_steps = 40
        max_steps = 1000
        if self.args.batch_size == 1:
            warmup_steps = 200

        cmd_args = ' ' + self.args.data_location + \
                   ' ' + self.args.checkpoint + ' --distance_metric 1' + \
                   ' --use_flipped_images' + ' --subtract_mean' + \
                   ' --use_fixed_image_standardization' + \
                   ' --num_inter_threads=' + \
                   str(self.args.num_inter_threads) + \
                   ' --num_intra_threads=' + \
                   str(self.args.num_intra_threads) + \
                   ' --lfw_batch_size=' + str(self.args.batch_size) + \
                   ' --lfw_pairs=' + self.args.lfw_pairs + \
                   ' --warmup_steps=' + str(warmup_steps) + \
                   ' --max_steps=' + str(max_steps)

        self.cmd = self.cmd + run_script + cmd_args

    def run(self):
        """run command to enable model benchmark or accuracy measurement"""

        original_dir = os.getcwd()
        os.chdir(self.args.model_source_dir)
        if self.cmd:
            self.run_command(self.cmd)
        os.chdir(original_dir)
