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

import os
import sys


class ModelInitializer:
    '''Add code here to detect the environment and set necessary variables
    before launching the model'''
    args = None
    custom_args = []

    def __init__(self, args, custom_args, platform_util):
        self.args = args
        self.custom_args = custom_args

        if self.args.verbose:
            print('Received these standard args: {}'.format(self.args))
            print('Received these custom args: {}'.format(self.custom_args))

        if args.mode == "inference":
            os.environ["OMP_NUM_THREADS"] = "1"

            if self.args.batch_size == -1:
                self.args.batch_size = 1
                if self.args.verbose:
                    print 'Setting batch_size to 1 since it is not supplied.'

            if self.args.batch_size == 1:
                if self.args.verbose:
                    print 'Running Wide_Deep model Inference in Latency mode'
            else:
                if self.args.verbose:
                    print 'Running Wide_Deep model Inference in ' \
                          'Throughput mode'

            # Select script based on batch size
            if self.args.batch_size == 1:
                executable = \
                  " classification/tensorflow/wide_deep/inference/" \
                  "fp32/wide_deep_inference_bs1_latency.py"
            else:
                executable = \
                  " classification/tensorflow/wide_deep/inference/" \
                  "fp32/wide_deep_inference.py"

        else:
            # TODO: Add support for training
            sys.exit("Training is currently not supported.")

        self.run_cmd = " OMP_NUM_THREADS=1" + \
                       " numactl --cpunodebind=0 --membind=0 " + \
                       " python " + executable + \
                       " --data_dir=" + self.args.data_location + \
                       " --model_dir=" + self.args.checkpoint + \
                       " --batch_size=" + str(self.args.batch_size)

    def run(self):
        if self.args.verbose:
            print("Run model here.")
        original_dir = os.getcwd()
        os.chdir(self.args.intelai_models)
        os.system(self.run_cmd)
        os.chdir(original_dir)
