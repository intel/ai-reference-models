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


os.environ["KMP_BLOCKTIME"] = "1"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine, compact"


class ModelInitializer(BaseModelInitializer):
    """ Model initializer for 3D UNet"""

    def __init__(self, args, custom_args=[], platform_util=None):
        self.args = args
        self.custom_args = custom_args
        self.platform_util = platform_util

        self.set_default_inter_intra_threads(platform_util)
        os.environ["OMP_NUM_THREADS"] = str(self.args.num_intra_threads)
        os.environ['KMP_HW_SUBSET'] = "28c,1T"
        script_path = os.path.join(self.args.intelai_models, self.args.mode,
                                   self.args.precision, "brats", "predict.py")

        # add numactl prefix to the command
        self.command_prefix = self.get_numactl_command(self.args.socket_id) + \
            "python " + script_path

        # add additional args to the command
        self.command_prefix += \
            " --inter {} --intra {} --nw 1 --nb 5 --bs {}".\
            format(self.args.num_inter_threads, self.args.num_intra_threads,
                   self.args.batch_size)

    def run(self):
        self.run_command(self.command_prefix)
