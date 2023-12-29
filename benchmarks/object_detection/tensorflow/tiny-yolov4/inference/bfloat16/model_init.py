#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from common.base_model_init import BaseModelInitializer
from common.base_model_init import set_env_var

import os
from argparse import ArgumentParser
import time


class ModelInitializer(BaseModelInitializer):
    """initialize mode and run benchmark"""

    def __init__(self, args, custom_args=[], platform_util=None):
        super(ModelInitializer, self).__init__(args, custom_args, platform_util)

        self.benchmark_command = ""
        if not platform_util:
            raise ValueError("Did not find any platform info.")

        # use default batch size if -1
        if self.args.batch_size == -1:
            self.args.batch_size = 1

        # set num_inter_threads and num_intra_threads
        # self.set_num_inter_intra_threads()

        arg_parser = ArgumentParser(description='Parse args')

        arg_parser.add_argument(
            '--kmp-blocktime', dest='kmp_blocktime',
            help='number of kmp block time',
            type=int, default=1)
        self.args = arg_parser.parse_args(self.custom_args, namespace=self.args)

        # Set KMP env vars, if they haven't already been set, but override the default KMP_BLOCKTIME value
        config_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        self.set_kmp_vars(config_file_path, kmp_blocktime=str(self.args.kmp_blocktime))

        set_env_var("OMP_NUM_THREADS", self.args.num_intra_threads)

        script_file = "infer_detections.py"

        benchmark_script = os.path.join(
            self.args.intelai_models, self.args.mode, script_file)

        self.benchmark_command = self.get_command_prefix(args.socket_id) + \
            self.python_exe + " " + benchmark_script

        if self.args.input_graph:
            self.benchmark_command += " --input-graph=" + self.args.input_graph

        # if the data location directory is not empty, then include the arg
        if self.args.data_location and os.listdir(self.args.data_location):
            self.benchmark_command += " --data-location=" + self.args.data_location + "/coco_val.record"

        # If batch size was specified
        if self.args.batch_size:
            self.benchmark_command += " --batch-size=" + str(self.args.batch_size)

        # Set parallelism thread values
        if self.args.num_inter_threads:
            self.benchmark_command += " --inter-op-parallelism-threads=" + str(self.args.num_inter_threads)
        if self.args.num_intra_threads:
            self.benchmark_command += " --intra-op-parallelism-threads=" + str(self.args.num_intra_threads)

        # Accuracy not enabled yet
        # if self.args.accuracy_only:
        #    self.benchmark_command += " --accuracy-only"

        if self.args.precision == 'bfloat16':
            self.benchmark_command += " --precision=bfloat16"

        # if output results is enabled, generate a results file name and pass it to the inference script
        if self.args.output_results:
            self.results_filename = "{}_{}_{}_results_{}.txt".format(
                self.args.model_name, self.args.precision, self.args.mode,
                time.strftime("%Y%m%d_%H%M%S", time.gmtime()))
            self.results_file_path = os.path.join(self.args.output_dir, self.results_filename)
            self.benchmark_command += " --results-file-path {}".format(self.results_file_path)

    def run(self):
        if self.benchmark_command:
            self.run_command(self.benchmark_command)
            if self.args.output_results:
                print("Inference results file in the output directory: {}".format(self.results_filename))
