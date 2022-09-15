#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
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
import sys

from common.base_model_init import BaseModelInitializer
from common.base_model_init import set_env_var

import argparse


class ModelInitializer(BaseModelInitializer):
    def run_inference_sanity_checks(self, args, custom_args):
        if not args.data_location:
            sys.exit("Please provide a path to the data directory via the "
                     "'--data-location' flag.")
        if args.socket_id == -1 and args.num_cores == -1:
            print("***Warning***: Running inference on all cores could degrade"
                  " performance. Pass a '--socket-id' to specify running on a"
                  " single socket instead.\n")

    def __init__(self, args, custom_args, platform_util):
        super(ModelInitializer, self).__init__(args, custom_args, platform_util)

        arg_parser = argparse.ArgumentParser(description='Parse additional args')

        self.additional_args, unknown_args = arg_parser.parse_known_args(custom_args)
        self.run_inference_sanity_checks(self.args, self.custom_args)

        # Set KMP env vars, if they haven't already been set
        config_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        self.set_kmp_vars(config_file_path)

        self.set_num_inter_intra_threads()

        set_env_var("OMP_NUM_THREADS", self.args.num_intra_threads)

        self.model_dir = os.path.join(self.args.intelai_models, self.args.mode, self.args.precision)

        train_script = os.path.join(self.model_dir, "train.py")

        # get command with numactl
        self.run_cmd = self.get_command_prefix(self.args.socket_id)

        self.run_cmd += "{0} {1}".format(self.python_exe, train_script)
        self.run_cmd += " --batch_size {0}".format(args.batch_size)
        self.run_cmd += " --data_type {0}".format(args.precision)
        self.run_cmd += " --num_inter_threads {0}".format(self.args.num_inter_threads)
        self.run_cmd += " --num_intra_threads {0}".format(self.args.num_intra_threads)
        self.run_cmd += " --data_location {0}".format(self.args.data_location)
        self.run_cmd += " --mode {0}".format("train")

        # self.run_cmd += " --timeline {0}".format(True)

    def run(self):
        self.run_command(self.run_cmd)
