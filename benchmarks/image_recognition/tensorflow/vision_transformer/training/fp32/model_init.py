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
    """Model Initializer for Vision Transformer training"""

    def __init__(self, args, custom_args=[], platform_util=None):
        super(ModelInitializer, self).__init__(
            args, custom_args, platform_util)

        self.benchmark_command = ""
        if not platform_util:
            raise ValueError("Did not find any platform info.")

        # use default batch size if -1
        if self.args.batch_size == -1:
            self.args.batch_size = 512

        # set num_inter_threads and num_intra_threads
        self.set_num_inter_intra_threads()

        arg_parser = ArgumentParser(description='Parse args')

        arg_parser.add_argument("--epochs", dest='epochs',
                                type=int, default=0,
                                help="number of epochs")
        arg_parser.add_argument("--model_dir", dest='model_dir',
                                type=str, default='/tmp/vit-training/',
                                help='Specify the location of the '
                                'output directory for logs and saved model')
        arg_parser.add_argument('--init-checkpoint', help=' Input pretrained model dir',
                                dest="init_checkpoint", required=True)
        arg_parser.add_argument('--kmp-blocktime', dest='kmp_blocktime',
                                help='number of kmp block time',
                                type=int, default=1)

        self.args = arg_parser.parse_args(self.custom_args, namespace=self.args)

        # Set KMP env vars, if they haven't already been set, but override the default KMP_BLOCKTIME value
        config_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        self.set_kmp_vars(config_file_path, kmp_blocktime=str(self.args.kmp_blocktime))

        if not self.args.gpu:
            set_env_var("OMP_NUM_THREADS", self.args.num_intra_threads)

        benchmark_script = os.path.join(
            self.args.intelai_models, self.args.mode, "vit_fine_tune.py")

        os.environ["PYTHONPATH"] = "{}:{}".format(os.environ["PYTHONPATH"],
                                                  os.path.join(self.args.intelai_models, self.args.mode))
        self.benchmark_command = self.get_command_prefix(args.socket_id) + \
            self.python_exe + " " + benchmark_script

        self.benchmark_command = \
            self.benchmark_command + \
            " --batch-size=" + str(self.args.batch_size) + \
            " --epochs=" + str(self.args.epochs) + \
            " --precision=" + str(self.args.precision) + \
            " --model-dir=" + str(self.args.model_dir)

        # if the data location is not empty, then include the arg
        if self.args.data_location and os.listdir(self.args.data_location):
            self.benchmark_command += " --data-location=" + self.args.data_location
        # if the initial checkpoint dir is not empty, then include the arg
        if self.args.init_checkpoint and os.listdir(self.args.init_checkpoint):
            self.benchmark_command += " --init-checkpoint=" + self.args.init_checkpoint

    def run(self):
        if self.benchmark_command:
            self.run_command(self.benchmark_command)
