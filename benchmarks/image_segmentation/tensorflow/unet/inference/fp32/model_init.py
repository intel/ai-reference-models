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

import argparse
import os
from common.base_model_init import BaseModelInitializer


class ModelInitializer(BaseModelInitializer):
    """ Model initializer for UNet FP32 inference """

    def parse_custom_args(self):
        if self.custom_args:
            parser = argparse.ArgumentParser()
            parser.add_argument("--checkpoint_name", default=None,
                                dest='checkpoint_name', type=str)
            self.args = parser.parse_args(self.custom_args,
                                          namespace=self.args)

    def __init__(self, args, custom_args=[], platform_util=None):
        super(ModelInitializer, self).__init__(args, custom_args, platform_util)

        self.parse_custom_args()
        self.set_num_inter_intra_threads()

        # Set KMP env vars, if they haven't already been set
        config_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        self.set_kmp_vars(config_file_path)

        # Get path to the inference script
        script_path = os.path.join(
            self.args.benchmark_dir, self.args.use_case, self.args.framework,
            self.args.model_name, self.args.mode, self.args.precision,
            "unet_infer.py")

        # Create the command prefix using numactl
        self.command_prefix = self.get_command_prefix(self.args.socket_id) +\
            "{} {}".format(self.python_exe, script_path)

        # Add batch size arg
        if self.args.batch_size != -1:
            self.command_prefix += " -bs {}".format(str(self.args.batch_size))

        # Add additional args
        checkpoint_path = os.path.join(self.args.checkpoint,
                                       self.args.checkpoint_name)
        self.command_prefix += " -cp {} --num_inter_threads {} " \
                               "--num_intra_threads {} -nw 80 -nb 400".\
            format(checkpoint_path, self.args.num_inter_threads,
                   self.args.num_intra_threads)

    def run(self):
        if self.command_prefix:
            self.run_command(self.command_prefix)
