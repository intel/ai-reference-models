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

import os

from common.base_model_init import BaseModelInitializer
from common.base_model_init import set_env_var


class ModelInitializer(BaseModelInitializer):
    '''Add code here to detect the environment and set necessary variables
    before launching the model'''

    def __init__(self, args, custom_args, platform_util):
        super(ModelInitializer, self).__init__(args, custom_args, platform_util)

        set_env_var("OMP_NUM_THREADS", "1")

        if args.batch_size == -1:
            args.batch_size = 1
            if args.verbose:
                print("Setting batch_size to 1 since it is not supplied.")

        if args.batch_size == 1:
            if args.verbose:
                print("Running Wide_Deep model Inference in Latency mode")
        else:
            if args.verbose:
                print("Running Wide_Deep model Inference in Throughput mode")

        executable = os.path.join(args.mode, args.precision,
                                  "wide_deep_inference.py")

        self.run_cmd = " OMP_NUM_THREADS=1" + \
                       " numactl --cpunodebind=0 --membind=0 " + \
                       self.python_exe + " " + executable + \
                       " --data_dir=" + self.args.data_location + \
                       " --model_dir=" + self.args.checkpoint + \
                       " --batch_size=" + str(self.args.batch_size)

    def run(self):
        original_dir = os.getcwd()
        os.chdir(self.args.intelai_models)
        self.run_command(self.run_cmd)
        os.chdir(original_dir)
