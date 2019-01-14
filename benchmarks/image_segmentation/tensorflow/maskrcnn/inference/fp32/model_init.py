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

os.environ["KMP_BLOCKTIME"] = "1"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine, compact, 1, 0"
os.environ["KMP_HW_SUBSET"] = "1T"


class ModelInitializer(BaseModelInitializer):
    """initialize model and run benchmark"""

    def __init__(self, args, custom_args=[], platform_util=None):
        self.args = args
        self.custom_args = custom_args
        self.platform_util = platform_util

        # set num_inter_threads and num_intra_threads
        self.set_default_inter_intra_threads(self.platform_util)

        benchmark_script = os.path.join(
            self.args.intelai_models, "coco.py")
        self.benchmark_command = self.get_numactl_command(args.socket_id) + \
            "python3 " + benchmark_script + " evaluate "

        os.environ["OMP_NUM_THREADS"] = str(self.args.num_intra_threads)

        self.benchmark_command = self.benchmark_command + \
            " --dataset=" + str(self.args.data_location) + \
            " --num_inter_threads " + str(self.args.num_inter_threads) + \
            " --num_intra_threads " + str(self.args.num_intra_threads) + \
            " --nw 5 --nb 50 --model=coco" + \
            " --infbs " + str(self.args.batch_size)

    def run(self):
        if self.benchmark_command:
            self.run_command(self.benchmark_command)
