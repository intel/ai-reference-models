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


class ModelInitializer(BaseModelInitializer):
    """Model initializer for 3D UNet"""

    def __init__(self, args, custom_args=[], platform_util=None):
        super(ModelInitializer, self).__init__(args, custom_args, platform_util)

        self.set_num_inter_intra_threads()
        set_env_var("OMP_NUM_THREADS", self.args.num_intra_threads)

        # Set KMP env vars, if they haven't already been set
        config_file_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "config.json"
        )
        self.set_kmp_vars(config_file_path)
        set_env_var("KMP_HW_SUBSET", "{}c,1T".format(self.args.num_intra_threads))
        script_path = os.path.join(
            self.args.intelai_models,
            self.args.mode,
            self.args.precision,
            "brats",
            "predict.py",
        )

        # add numactl prefix to the command
        self.command_prefix = (
            self.get_command_prefix(self.args.socket_id) + "python " + script_path
        )

        # add additional args to the command
        self.command_prefix += " --inter {} --intra {} --nw 1 --nb 5 --bs {}".format(
            self.args.num_inter_threads,
            self.args.num_intra_threads,
            self.args.batch_size,
        )

    def run(self):
        self.run_command(self.command_prefix)
