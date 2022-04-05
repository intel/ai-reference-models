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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from common.base_model_init import BaseModelInitializer
from common.base_model_init import set_env_var

import os
from argparse import ArgumentParser


class ResNet50ModelInitializer(BaseModelInitializer):
    """initialize mode and run benchmark"""

    def __init__(self, args, custom_args=[], platform_util=None):
        super(ResNet50ModelInitializer, self).__init__(
            args, custom_args, platform_util)

        self.benchmark_command = ""
        if not platform_util:
            raise ValueError("Did not find any platform info.")

        # use default batch size if -1
        if self.args.batch_size == -1:
            self.args.batch_size = 64

        # set num_inter_threads and num_intra_threads
        self.set_num_inter_intra_threads()

        arg_parser = ArgumentParser(description='Parse args')

        arg_parser.add_argument("--steps", dest='steps',
                                type=int, default=112590,
                                help="number of steps")
        arg_parser.add_argument("--train_epochs", dest='trainepochs',
                                type=int, default=72,
                                help="number of epochs")
        arg_parser.add_argument("--epochs_between_evals", dest='epochsbtwevals',
                                type=int, default=1,
                                help="number of epochs between eval")
        arg_parser.add_argument('--kmp-blocktime', dest='kmp_blocktime',
                                help='number of kmp block time',
                                type=int, default=1)

        self.args = arg_parser.parse_args(self.custom_args, namespace=self.args)

        # Set KMP env vars, if they haven't already been set, but override the default KMP_BLOCKTIME value
        config_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        self.set_kmp_vars(config_file_path, kmp_blocktime=str(self.args.kmp_blocktime))

        set_env_var("OMP_NUM_THREADS", self.args.num_intra_threads)

        benchmark_script = os.path.join(
            self.args.intelai_models, self.args.mode,
            "mlperf_resnet/imagenet_main.py")

        # We need to change directory to model source to avoid python
        # module name conflicts.
        # self.benchmark_command = "cd " + self.args.model_source_dir + \
        #    "/models && " + self.get_command_prefix(args.socket_id) + \
        #    self.python_exe + " " + benchmark_script

        os.environ["PYTHONPATH"] = "{}:{}".format(os.environ["PYTHONPATH"],
                                                  os.path.join(self.args.intelai_models, self.args.mode))
        self.benchmark_command = self.get_command_prefix(args.socket_id) + \
            self.python_exe + " " + benchmark_script

        # Model requires random_seed. Just setting it to a random value.
        random_seed = 2
        self.benchmark_command = \
            self.benchmark_command + \
            " " + str(random_seed) + \
            " --batch_size=" + str(self.args.batch_size) + \
            " --max_train_steps=" + str(self.args.steps) + \
            " --train_epochs=" + str(self.args.trainepochs) + \
            " --epochs_between_evals=" + str(self.args.epochsbtwevals) + \
            " --inter_op_parallelism_threads " + str(self.args.num_inter_threads) + \
            " --intra_op_parallelism_threads " + str(self.args.num_intra_threads) + \
            " --version 1 --resnet_size 50 --data_format=channels_last"

        # if the data location and checkpoint directory is not empty, then include the arg
        if self.args.data_location and os.listdir(self.args.data_location):
            self.benchmark_command += " --data_dir=" + \
                                      self.args.data_location
        if self.args.checkpoint:
            self.benchmark_command += " --model_dir=" + \
                                      self.args.checkpoint

    def run(self):
        if self.benchmark_command:
            self.run_command(self.benchmark_command)
