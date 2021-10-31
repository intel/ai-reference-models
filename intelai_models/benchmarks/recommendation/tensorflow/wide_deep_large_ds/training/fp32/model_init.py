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

#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import argparse
from common.base_model_init import BaseModelInitializer


class ModelInitializer(BaseModelInitializer):
    """Model initializer for Wide and deep large dataset FP32 inference"""

    def __init__(self, args, custom_args=[], platform_util=None):
        super(ModelInitializer, self).__init__(args, custom_args, platform_util)

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--steps", dest='steps',
                            type=int, default=0,
                            help="number of train steps")
        self.args = parser.parse_args(self.custom_args,
                                      namespace=self.args)

    def run_benchmark(self):
        benchmark_script = os.path.join(self.args.intelai_models,
                                        self.args.mode, "train.py")
        script_args_list = ["batch_size", "data_location", "steps", "output_dir", "checkpoint"]
        command_prefix = self.get_command_prefix(-1)
        cmd_prefix = command_prefix + self.python_exe + " " + benchmark_script
        cmd = self.add_args_to_command(cmd_prefix, script_args_list)
        self.run_command(cmd)

    def run(self):
        # Parse custom arguments and append to self.args
        self.parse_args()
        self.run_benchmark()
