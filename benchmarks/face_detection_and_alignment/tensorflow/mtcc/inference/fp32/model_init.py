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

import os

from common.base_model_init import BaseModelInitializer
from common.base_model_init import set_env_var


class ModelInitializer (BaseModelInitializer):

    def __init__(self, args, custom_args, platform_util=None):
        super(ModelInitializer, self).__init__(args, custom_args, platform_util)

        # set num_inter_threads and num_intra_threads
        self.set_num_inter_intra_threads()

        # Set KMP env vars, if they haven't already been set
        self.set_kmp_vars()

        set_env_var("OMP_NUM_THREADS", self.args.num_intra_threads)

        benchmark_script = os.path.join(
            self.args.intelai_models, self.args.mode, self.args.precision,
            "one_image_test.py")
        self.command_prefix = \
            self.get_numactl_command(self.args.socket_id) + \
            "{} ".format(self.python_exe) + benchmark_script

        self.run_cmd = \
            self.command_prefix + \
            " --num_inter_threads " + str(self.args.num_inter_threads) + \
            " --num_intra_threads " + str(self.args.num_intra_threads) + \
            " -ckpt " + self.args.checkpoint + \
            " -dl " + self.args.data_location

    def run(self):
        self.run_command(self.run_cmd)
