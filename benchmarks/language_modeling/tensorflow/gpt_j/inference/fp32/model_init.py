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

#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from common.base_model_init import BaseModelInitializer
from common.base_model_init import set_env_var

import os
from argparse import ArgumentParser


class ModelInitializer(BaseModelInitializer):
    """initialize mode and run benchmark"""

    def __init__(self, args, custom_args=[], platform_util=None):
        super(ModelInitializer, self).__init__(args, custom_args, platform_util)

        self.benchmark_command = ""
        if not platform_util:
            raise ValueError("Did not find any platform info.")

        # set num_inter_threads and num_intra_threads
        self.set_num_inter_intra_threads()

        arg_parser = ArgumentParser(description='Parse args')

        arg_parser.add_argument(
            '--kmp-blocktime', dest='kmp_blocktime',
            help='number of kmp block time',
            type=int, default=1)

        arg_parser.add_argument(
            '--data_model_dir', dest='data_model_dir',
            help='Location to store data and pretrained model.',
            type=str, default=None)

        arg_parser.add_argument(
            '--max_output_tokens', dest='max_output_tokens',
            help='Maximum tokens to output.',
            type=int, default=32)

        arg_parser.add_argument(
            '--input_tokens', dest='input_tokens',
            help='Input tokens.',
            type=int, default=32)

        arg_parser.add_argument(
            '--skip_rows', dest='skip_rows',
            help='Skip rows for latency use-case',
            type=int, default=0)

        self.args = arg_parser.parse_args(self.custom_args, namespace=self.args)

        # Set KMP env vars, if they haven't already been set, but override the default KMP_BLOCKTIME value
        config_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        self.set_kmp_vars(config_file_path, kmp_blocktime=str(self.args.kmp_blocktime))

        set_env_var("OMP_NUM_THREADS", self.args.num_intra_threads)

        if self.args.benchmark_only:
            benchmark_script = os.path.join(
                self.args.intelai_models, self.args.mode,
                "run_benchmark.py")
        else:
            benchmark_script = os.path.join(
                self.args.intelai_models, self.args.mode,
                "run_eval.py")

        self.benchmark_command = self.get_command_prefix(args.socket_id) + \
            self.python_exe + " " + benchmark_script

        if self.args.benchmark_only:
            if self.args.batch_size:
                self.benchmark_command = self.benchmark_command + \
                    " --batch_size={}".format(
                        self.args.batch_size
                    )

            if self.args.input_tokens:
                self.benchmark_command = self.benchmark_command + \
                    " --input_tokens={}".format(
                        self.args.input_tokens
                    )

            if self.args.max_output_tokens:
                self.benchmark_command = self.benchmark_command + \
                    " --max_output_tokens={}".format(
                        self.args.max_output_tokens
                    )

            if self.args.skip_rows == 1:
                self.benchmark_command = self.benchmark_command + \
                    " --skip_rows"

        self.benchmark_command = \
            self.benchmark_command + \
            " --model_name_or_path EleutherAI/gpt-j-6B"

        self.benchmark_command = \
            self.benchmark_command + \
            " --dataset_name EleutherAI/lambada_openai"

        self.benchmark_command = \
            self.benchmark_command + \
            " --precision fp32"

        if self.args.checkpoint and os.listdir(self.args.checkpoint):
            self.benchmark_command += " --checkpoint=" + \
                                      self.args.checkpoint

        if self.args.output_dir and os.listdir(self.args.output_dir):
            self.benchmark_command += " --output_dir=" + \
                                      self.args.output_dir

    def run(self):
        if self.benchmark_command:
            self.run_command(self.benchmark_command)
