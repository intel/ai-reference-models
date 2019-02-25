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
from common.base_model_init import BaseModelInitializer
from common.base_model_init import set_env_var

import os
import time


class ModelInitializer(BaseModelInitializer):
    """initialize mode and run benchmark"""

    def __init__(self, args, custom_args, platform_util=None):
        super(ModelInitializer, self).__init__(args, custom_args, platform_util)

        self.benchmark_command = ""  # use default batch size if -1
        self.results_file_path = ""

        if self.args.batch_size == -1:
            self.args.batch_size = 1024

        # get number of cores if there is no core info passed in.
        if self.args.num_cores == -1:
            self.args.num_cores = self.platform_util.num_cores_per_socket()

        num_of_parallel_batches = self.platform_util.num_cores_per_socket()
        kmp_blocktime = "0"
        # Set KMP env vars, if they haven't already been set
        self.set_kmp_vars(kmp_settings="1", kmp_blocktime=kmp_blocktime,
                          kmp_affinity="noverbose,warnings,respect,granularity=core,none")
        # set num_inter_threads and num_intra_threads
        self.set_num_inter_intra_threads(num_inter_threads=self.args.num_cores, num_intra_threads=1)

        benchmark_script = os.path.join(
            self.args.intelai_models,
            self.args.mode,
            "inference.py")

        self.benchmark_command = self.get_numactl_command(self.args.socket_id)\
            + "python " + benchmark_script

        set_env_var("OMP_NUM_THREADS", self.args.num_intra_threads)
        self.benchmark_command += " --input-graph=" + \
                                  self.args.input_graph + \
                                  " --inter-op-parallelism-threads=" + \
                                  str(self.args.num_inter_threads) + \
                                  " --intra-op-parallelism-threads=" + \
                                  str(self.args.num_intra_threads) + \
                                  " --omp-num-threads=" + \
                                  str(self.args.num_intra_threads) + \
                                  " --batch-size=" + \
                                  str(self.args.batch_size) + \
                                  " --num-of-parallel-batches=" + \
                                  str(num_of_parallel_batches) + \
                                  " --kmp-blocktime=" + \
                                  str(kmp_blocktime)

        # if the data location directory is not empty, then include the arg
        if self.args.data_location:
            self.benchmark_command += " --datafile-path=" + \
                                      self.args.data_location
            # if output results is enabled, generate a results file name and pass it to the inference script
            if self.args.output_results:
                self.results_filename = "{}_{}_{}_results_{}.txt".format(
                    self.args.model_name, self.args.precision, self.args.mode,
                    time.strftime("%Y%m%d_%H%M%S", time.gmtime()))
                self.results_file_path = os.path.join(self.args.output_dir, self.results_filename)
                self.benchmark_command += " --results-file-path {}".format(self.results_file_path)

    def run(self):
        if self.benchmark_command:
            print(self.benchmark_command)
            self.run_command(self.benchmark_command)
            if self.results_file_path:
                print("Inference results file in the output directory: {}".format(self.results_filename))
