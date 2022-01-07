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

import os
import sys
import argparse

from common.base_model_init import BaseModelInitializer, set_env_var


class ModelInitializer(BaseModelInitializer):
    command = []
    RFCN_PERF_SCRIPT = "run_rfcn_inference.py"
    RFCN_ACCURACY_SCRIPT = "coco_mAP.sh"
    perf_script_path = ""
    accuracy_script_path = ""

    def __init__(self, args, custom_args=[], platform_util=None):
        super(ModelInitializer, self).__init__(args, custom_args, platform_util)
        self.perf_script_path = os.path.join(
            self.args.intelai_models, self.args.mode, self.args.precision,
            self.RFCN_PERF_SCRIPT)
        self.accuracy_script_path = os.path.join(
            self.args.intelai_models, self.args.mode, self.args.precision,
            self.RFCN_ACCURACY_SCRIPT)

        # remove intelai models path, so that imports don't conflict
        if "MOUNT_BENCHMARK" in os.environ and \
                os.environ["MOUNT_BENCHMARK"] in sys.path:
            sys.path.remove(os.environ["MOUNT_BENCHMARK"])
        if self.args.intelai_models in sys.path:
            sys.path.remove(self.args.intelai_models)

        self.parse_args()

        # Set KMP env vars, if they haven't already been set
        config_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        self.set_kmp_vars(config_file_path)

        # Set num_inter_threads and num_intra_threads
        self.set_num_inter_intra_threads()

    def parse_args(self):
        if self.custom_args:
            parser = argparse.ArgumentParser()
            mutex_group = parser.add_mutually_exclusive_group()
            mutex_group.add_argument("-x", "--number_of_steps",
                                     help="Run for n number of steps",
                                     type=int, default=None)
            mutex_group.add_argument(
                "-v", "--visualize",
                help="Whether to visualize the output image",
                action="store_true")
            parser.add_argument(
                "-t", "--timeline",
                help="Output file name for TF timeline",
                type=str, default=None)
            parser.add_argument("-e", "--evaluate_tensor",
                                help="Full tensor name to evaluate",
                                type=str, default=None)
            parser.add_argument("-p", "--print_accuracy",
                                help="Print accuracy results",
                                action="store_true")
            parser.add_argument("-q", "--split",
                                help="Location of accuracy data",
                                type=str, default=None)
            self.args = parser.parse_args(self.custom_args,
                                          namespace=self.args)
            self.validate_args()
        else:
            raise ValueError("Custom parameters are missing...")

    def validate_args(self):
        if not (self.args.batch_size == -1 or self.args.batch_size == 1):
            raise ValueError(
                "Batch Size specified: {}. R-FCN inference only supports "
                "batch size = 1".format(self.args.batch_size))

        if not os.path.exists(self.perf_script_path):
            raise ValueError("Unable to locate the R-FCN perf script: {}".
                             format(self.perf_script_path))

        if not os.path.exists(self.accuracy_script_path):
            raise ValueError("Unable to locate the R-FCN accuracy script: "
                             "{}".format(self.accuracy_script_path))

        if not self.args.model_source_dir or not os.path.isdir(
                self.args.model_source_dir):
            raise ValueError("Unable to locate TensorFlow models at {}".
                             format(self.args.model_source_dir))

    def run_perf_command(self):
        # Get the command previx, but numactl is added later in run_perf_command()
        self.command.append(self.get_command_prefix(self.args.socket_id, numactl=False))

        set_env_var("OMP_NUM_THREADS", self.args.num_intra_threads)

        num_numas = self.platform_util.num_numa_nodes
        if self.args.socket_id != -1 and num_numas > 0:
            self.command.append("numactl")
            if self.args.socket_id:
                socket_id = self.args.socket_id
            else:
                socket_id = "0"

            if self.args.num_cores != -1:
                self.command.append("-C")
                cpuid = "+0"
                i = 1
                while i < self.args.num_cores:
                    cpuid += ',' + str(i)
                    i += i
                self.command.append(cpuid)

            self.command.append("-N")
            self.command.append("{}".format(socket_id))
            self.command.append("-m")
            self.command.append("{}".format(socket_id))

        self.command += (self.python_exe, self.perf_script_path)
        self.command += ("-m", self.args.model_source_dir)
        self.command += ("-g", self.args.input_graph)
        self.command += ("--num-intra-threads", str(self.args.num_intra_threads))
        self.command += ("--num-inter-threads", str(self.args.num_inter_threads))
        if self.args.number_of_steps:
            self.command += ("-x", "{}".format(self.args.number_of_steps))
        if self.args.visualize:
            self.command += ("-v")
        if self.args.timeline:
            self.command += ("-t", self.args.timeline)
        if self.args.data_location:
            self.command += ("-d", self.args.data_location)
        if self.args.evaluate_tensor:
            self.command += ("-e", self.args.evaluate_tensor)
        if self.args.print_accuracy:
            self.command += ("-p")
        self.run_command(" ".join(self.command))

    def run_accuracy_command(self):
        # already validated by parent
        self.command = self.get_command_prefix(self.args.socket_id, numactl=False)
        os.environ["FROZEN_GRAPH"] = "{}".format(self.args.input_graph)

        if self.args.data_location and os.path.exists(
                self.args.data_location):
            os.environ["TF_RECORD_FILE"] = "{}".format(self.args.data_location)
        else:
            raise ValueError(
                "Unable to locate the coco data record file at {}".format(
                    self.args.tf_record_file))

        if self.args.split:
            os.environ["SPLIT"] = "{}".format(self.args.split)
        else:
            raise ValueError("Must specify SPLIT parameter")

        os.environ["TF_MODELS_ROOT"] = "{}".format(self.args.model_source_dir)

        self.command += "bash " + self.accuracy_script_path
        self.run_command(self.command)

    def run(self):
        # TODO: make it a property in PlatformUtils (platform_util.os_type) to get the host OS.
        # We already do the OS check there to see if it's one that we support.
        if os.environ.get('OS', '') == 'Windows_NT':
            os.environ["PYTHONPATH"] = "{};{};{}".format(
                os.path.join(self.args.model_source_dir, "research"),
                os.path.join(self.args.model_source_dir, "research", "slim"),
                os.environ["PYTHONPATH"])
        # Run script from the tensorflow models research directory
        original_dir = os.getcwd()
        os.chdir(os.path.join(self.args.model_source_dir, "research"))
        if self.args.accuracy_only:
            self.run_accuracy_command()
        else:
            self.run_perf_command()
        os.chdir(original_dir)
