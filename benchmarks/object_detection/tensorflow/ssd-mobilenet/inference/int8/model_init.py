#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Intel Corporation
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

import os
import sys

from common.base_model_init import BaseModelInitializer, set_env_var


class ModelInitializer(BaseModelInitializer):
    # SSD-MobileNet Int8 inference model initialization
    args = None
    custom_args = []

    def __init__(self, args, custom_args=[], platform_util=None):
        super(ModelInitializer, self).__init__(args, custom_args, platform_util)
        self.set_kmp_vars(kmp_blocktime="0")

        # set num_inter_threads and num_intra_threads (override inter threads to 2)
        self.set_num_inter_intra_threads(num_inter_threads=2)

        # remove intelai models path, so that imports don't conflict
        if "MOUNT_BENCHMARK" in os.environ and \
                os.environ["MOUNT_BENCHMARK"] in sys.path:
            sys.path.remove(os.environ["MOUNT_BENCHMARK"])
        if self.args.intelai_models in sys.path:
            sys.path.remove(self.args.intelai_models)
        threads_per_socket = platform_util.num_cores_per_socket * \
            platform_util.num_threads_per_core

        if self.args.benchmark_only:
            benchmark_script = os.path.join(
                self.args.intelai_models, self.args.mode, self.args.precision,
                "run_frozen_graph_ssdmob.py")
            self.command_prefix = self.get_numactl_command(self.args.socket_id) + \
                "{} {}".format(self.python_exe, benchmark_script)
            set_env_var("OMP_NUM_THREADS", self.args.num_intra_threads)

            self.command_prefix = "{} -g {} -n 5000 -d {} --num-inter-threads {} --num-intra-threads {}".format(
                self.command_prefix, self.args.input_graph, self.args.data_location,
                self.args.num_inter_threads, self.args.num_intra_threads)

            if self.args.socket_id != -1:
                self.command_prefix += " -x"
        else:
            set_env_var("OMP_NUM_THREADS", threads_per_socket)
            accuracy_script = os.path.join(
                self.args.intelai_models, self.args.mode, self.args.precision,
                "coco_int8.sh")
            self.command_prefix = "sh {} {} {}/coco_val.record".format(
                accuracy_script, self.args.input_graph,
                self.args.data_location)

    def run(self):
        # Run script from the tensorflow models research directory
        original_dir = os.getcwd()
        os.chdir(os.path.join(self.args.model_source_dir, "research"))
        self.run_command(self.command_prefix)
        os.chdir(original_dir)
