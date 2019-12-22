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

import argparse
import os
import sys

from common.base_model_init import BaseModelInitializer
from common.base_model_init import set_env_var


class ModelInitializer(BaseModelInitializer):
    accuracy_script = "coco_mAP.sh"
    accuracy_script_path = ""

    def run_inference_sanity_checks(self, args, custom_args):
        if args.batch_size != -1 and args.batch_size != 1:
            sys.exit("R-FCN inference supports 'batch-size=1' " +
                     "only, please modify via the '--batch_size' flag.")

    def __init__(self, args, custom_args, platform_util):
        super(ModelInitializer, self).__init__(args, custom_args, platform_util)

        self.accuracy_script_path = os.path.join(
            self.args.intelai_models, self.args.mode, self.args.precision,
            self.accuracy_script)
        self.benchmark_script = os.path.join(
            self.args.intelai_models, self.args.mode,
            self.args.precision, "eval.py")

        # Set KMP env vars, if they haven't already been set
        config_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        self.set_kmp_vars(config_file_path)

        # set num_inter_threads and num_intra_threds
        self.set_num_inter_intra_threads(num_inter_threads=self.args.num_inter_threads,
                                         num_intra_threads=self.args.num_intra_threads)
        self.omp_num_threads = self.args.num_intra_threads if self.args.num_cores == -1 else self.args.num_cores
        set_env_var("OMP_NUM_THREADS", self.omp_num_threads)

        self.parse_custom_args()
        self.run_inference_sanity_checks(self.args, self.custom_args)
        self.research_dir = os.path.join(self.args.model_source_dir,
                                         "research")

    def run_benchmark(self):
        command_prefix = self.get_command_prefix(self.args.socket_id) + \
            self.python_exe + " " + self.benchmark_script

        # set num_inter_threads and num_intra_threads
        self.set_num_inter_intra_threads()

        if self.args.socket_id == -1:
            if self.args.num_cores == 1:
                command_prefix = "taskset -c 0 " + \
                                 command_prefix
                self.args.num_intra_threads = 1
            else:
                command_prefix = "taskset -c 0-" + \
                                 str(self.args.num_cores - 1) + \
                                 " " + command_prefix

        config_file_path = os.path.join(self.args.checkpoint,
                                        self.args.config_file)

        run_cmd = command_prefix + \
            " --inter_op " + str(self.args.num_inter_threads) + \
            " --intra_op " + str(self.args.num_intra_threads) + \
            " --omp " + str(self.omp_num_threads) + \
            " --pipeline_config_path " + config_file_path + \
            " --checkpoint_dir " + str(self.args.checkpoint) + \
            " --eval_dir " + self.research_dir + \
            "/object_detection/models/rfcn/eval " + \
            " --logtostderr " + \
            " --blocktime=0 " + \
            " --run_once=True"
        self.run_command(run_cmd)

    def parse_custom_args(self):
        if self.custom_args:
            parser = argparse.ArgumentParser()
            parser.add_argument("--config_file", default=None,
                                dest="config_file", type=str)
            parser.add_argument("-q", "--split",
                                help="Location of accuracy data",
                                type=str, default=None)
            self.args = parser.parse_args(self.custom_args,
                                          namespace=self.args)

    def run_accuracy_command(self):
        if not os.path.exists(self.accuracy_script_path):
            raise ValueError("Unable to locate the R-FCN accuracy script: {}".format(self.accuracy_script_path))
        if not self.args.data_location or not os.path.exists(self.args.data_location):
            raise ValueError("Unable to locate the coco data record file at {}".format(self.args.tf_record_file))
        if not self.args.split:
            raise ValueError("Must specify SPLIT parameter")

        command = self.get_command_prefix(self.args.socket_id, numactl=False)
        command += " {} {} {} {} {}".format(self.accuracy_script_path,
                                            self.args.input_graph,
                                            self.args.data_location,
                                            self.args.model_source_dir,
                                            self.args.split)
        self.run_command(command)

    def run(self):
        original_dir = os.getcwd()
        os.chdir(self.research_dir)
        if self.args.accuracy_only:
            self.run_accuracy_command()
        else:
            self.run_benchmark()
        os.chdir(original_dir)
