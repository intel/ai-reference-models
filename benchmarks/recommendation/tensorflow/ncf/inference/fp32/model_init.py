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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

os.environ["KMP_BLOCKTIME"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
os.environ["KMP_SETTINGS"] = "1"


class ModelInitializer:
    """initialize mode and run benchmark"""

    def __init__(self, args, custom_args=[], platform_util=None):
        self.args = args
        self.custom_args = custom_args
        self.platform_util = platform_util
        self.benchmark_command = ''

        if self.args.verbose:
            print('Received these standard args: {}'.format(self.args))
            print('Received these custom args: {}'.format(self.custom_args))

        # use default batch size if -1
        if self.args.batch_size == -1:
            self.args.batch_size = 256

        if self.args.single_socket:
            self.args.num_inter_threads = 1
            self.args.num_intra_threads = int(
                self.platform_util.num_cores_per_socket() / 2) \
                if self.args.num_cores == -1 else self.args.num_cores
        else:
            self.args.num_inter_threads = self.platform_util.num_cpu_sockets()

            if self.args.num_cores == -1:
                self.args.num_intra_threads = \
                    int(self.platform_util.num_cores_per_socket() *
                        self.platform_util.num_cpu_sockets())
            else:
                self.args.num_intra_threads = self.args.num_cores

        if self.args.mode == "inference":
            benchmark_script = os.path.join(
                self.args.intelai_models, self.args.mode, self.args.platform,
                "ncf_main.py")

            self.benchmark_command = "python " + benchmark_script

            if self.args.single_socket:
                socket_id_str = str(self.args.socket_id)

            self.benchmark_command = 'numactl --cpunodebind=' + socket_id_str \
                                     + ' --membind=' + socket_id_str + ' ' + \
                                     self.benchmark_command

            os.environ["OMP_NUM_THREADS"] = str(self.args.num_intra_threads)

            self.benchmark_command = self.benchmark_command + \
                ' --data_dir=' + str(args.data_location) + \
                ' --model_dir=' + str(args.checkpoint) + \
                ' --intra_op_parallelism_threads=' + str(
                    self.args.num_intra_threads) + \
                ' --inter_op_parallelism_threads=' + str(
                    self.args.num_inter_threads) + \
                ' --batch_size=' + str(self.args.batch_size) + \
                ' --inference_only'
        else:
            print('Training is not supported currently.')

    def run(self):
        if self.benchmark_command:
            if self.args.verbose:
                print("Run model here.", self.benchmark_command)
            os.system(self.benchmark_command)
