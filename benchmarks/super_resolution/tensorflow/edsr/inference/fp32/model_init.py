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

import argparse
import os
from common.base_model_init import BaseModelInitializer
from common.base_model_init import set_env_var


class ModelInitializer(BaseModelInitializer):
    """ Model initializer for EDSR inference """

    def __init__(self, args, custom_args=[], platform_util=None):
        super(ModelInitializer, self).__init__(args, custom_args, platform_util)

        # use default batch size if -1
        if self.args.batch_size == -1:
            self.args.batch_size = 32

        # Set KMP env vars, if they haven't already been set
        config_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        self.set_kmp_vars(config_file_path)

        """ # set num_inter_threads and num_intra_threads (override inter threads to 2)
        self.set_num_inter_intra_threads(num_inter_threads=2) """

        script_name = "accuracy.py" if self.args.accuracy_only \
            else "benchmark.py"
        script_path = os.path.join(
            self.args.intelai_models, self.args.mode, self.args.precision, script_name)
        self.command_prefix = "{} {}".format(self.python_exe, script_path)

        num_numas = self.platform_util.num_numa_nodes
        if self.args.socket_id != -1 and num_numas > 0:
            self.command_prefix = "numactl --cpunodebind={} -l {}".format(
                str(self.args.socket_id), self.command_prefix)

        set_env_var("OMP_NUM_THREADS", self.args.num_intra_threads)

        self.parse_args()

        if not self.args.accuracy_only:
            # add args for the benchmark script
            script_args_list = [
                "input_graph", "batch_size", "input_layer", "output_layer",
                "num_inter_threads", "num_intra_threads", "warmup_steps", "steps", "precision", "use_real_data"]
            self.command_prefix = self.add_args_to_command(
                self.command_prefix, script_args_list)
        else:
            # add args for the accuracy script
            script_args_list = [
                "input_graph", "batch_size", "input_layer",
                "output_layer", "num_inter_threads", "num_intra_threads", "precision"]
            self.command_prefix = self.add_args_to_command(
                self.command_prefix, script_args_list)

    def parse_args(self):
        if self.custom_args is None:
            return

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--warmup_steps", dest="warmup_steps",
            help="number of warmup steps",
            type=int, default=10)
        parser.add_argument(
            "--steps", dest="steps",
            help="number of steps",
            type=int, default=50)
        parser.add_argument(
            "--input_layer", dest="input_layer",
            help="name of input layer",
            type=str, default="IteratorGetNext")
        parser.add_argument(
            "--output_layer", dest="output_layer",
            help="name of output layer",
            type=str, default="NHWC_output")
        parser.add_argument(
            "--use_real_data", dest="use_real_data",
            help="Specify if DIV2K dataset to be used for benchmarking.",
            type=bool, default=False
        )

        self.args = parser.parse_args(self.custom_args, namespace=self.args)

    def run(self):
        self.run_command(self.command_prefix)
