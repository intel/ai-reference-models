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
from common.base_model_init import BaseModelInitializer

import os


class ModelInitializer(BaseModelInitializer):
    """initialize model and run benchmark"""

    def __init__(self, args, custom_args=[], platform_util=None):
        super(ModelInitializer, self).__init__(args, custom_args, platform_util)

        self.benchmark_command = ""
        if not platform_util:
            raise ValueError("Did not find any platform info.")
        self.set_num_inter_intra_threads()

        model_script = os.path.join(self.args.intelai_models,
                                    self.args.mode, self.args.precision, "coco.py")

        # Set KMP env vars, if they haven't already been set
        config_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        self.set_kmp_vars(config_file_path)

        os.environ["OMP_NUM_THREADS"] = str(self.args.num_intra_threads)

        model_args = " --dataset=" + str(self.args.data_location) + \
            " --num_inter_threads " + str(self.args.num_inter_threads) + \
            " --num_intra_threads " + str(self.args.num_intra_threads) + \
            " --nw 5 --nb 50 --model=coco" + \
            " --infbs " + str(self.args.batch_size)

        self.benchmark_command = self.get_command_prefix(args.socket_id) + \
            self.python_exe + " " + model_script + " evaluate " + model_args

    def run(self):
        if self.benchmark_command:
            self.run_command(self.benchmark_command)
