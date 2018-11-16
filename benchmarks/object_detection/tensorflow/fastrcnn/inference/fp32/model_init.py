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

import argparse
import os
import sys

os.environ["KMP_BLOCKTIME"] = "1"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"


class ModelInitializer:
    args = None

    def run_inference_sanity_checks(self, args, custom_args):

        if args.batch_size != -1 and args.batch_size != 1:
            sys.exit('Fast R-CNN inference supports `batch-size=1` ' +
                     'only, please modify via the \`--batch_size\` flag.')

    def __init__(self, args, custom_args, platform_util=None):
        self.args = args
        self.custom_args = custom_args
        self.platform_util = platform_util

        if self.args.verbose:
            print('Received these standard args: {}'.format(self.args))
            print('Received these custom args: {}'.format(self.custom_args))

        self.parse_custom_args()
        if args.mode == "inference":
            self.run_inference_sanity_checks(self.args, self.custom_args)

            benchmark_script = os.path.join(
                self.args.intelai_models, self.args.mode, self.args.platform,
                "eval.py")
            self.command_prefix = "python " + benchmark_script
            if self.args.single_socket:
                self.args.num_inter_threads = 1
                self.args.num_intra_threads = \
                    self.platform_util.num_cores_per_socket()
                self.command_prefix = \
                    'numactl --cpunodebind=0 --membind=0 ' + \
                    self.command_prefix

            os.environ["OMP_NUM_THREADS"] = \
                str(self.args.num_intra_threads)
            self.research_dir = os.path.join(self.args.model_source_dir,
                                             "research")
            config_file_path = os.path.join(self.args.checkpoint,
                                            self.args.config_file)
            self.run_cmd = \
                self.command_prefix + \
                " --num_inter_threads " + \
                str(self.args.num_inter_threads) + \
                " --num_intra_threads " + \
                str(self.args.num_intra_threads) + \
                " --pipeline_config_path " + \
                config_file_path + \
                " --checkpoint_dir " + \
                str(args.checkpoint) + \
                " --eval_dir " + self.research_dir + \
                "/object_detection/log/eval"

        else:
            # TODO: Add training commands
            sys.exit("Training is currently not supported.")

    def parse_custom_args(self):
        if self.custom_args:
            parser = argparse.ArgumentParser()
            parser.add_argument("--config_file", default=None,
                                dest='config_file', type=str)
            self.args = parser.parse_args(self.custom_args,
                                          namespace=self.args)

    def run(self):
        if self.args.verbose: print("Run model here.")
        original_dir = os.getcwd()
        os.chdir(self.research_dir)
        print("current directory: {}".format(os.getcwd()))
        print("Running: " + str(self.run_cmd))
        os.system(self.run_cmd)
        os.chdir(original_dir)
