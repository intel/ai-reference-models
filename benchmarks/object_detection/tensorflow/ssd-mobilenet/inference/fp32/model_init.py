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

#

import os

from common.base_model_init import BaseModelInitializer, set_env_var


class ModelInitializer(BaseModelInitializer):
    # SSD-MobileNet inference model initialization
    args = None
    custom_args = []

    def __init__(self, args, custom_args=[], platform_util=None):
        super(ModelInitializer, self).__init__(args, custom_args, platform_util)

        # Set the num_inter_threads and num_intra_threads
        # if user did not provide then default value based on platform will be set
        self.set_num_inter_intra_threads(self.args.num_inter_threads,
                                         self.args.num_intra_threads)

        # Set KMP env vars, if they haven't already been set
        config_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        self.set_kmp_vars(config_file_path)

        benchmark_script = os.path.join(self.args.intelai_models, self.args.mode,
                                        "infer_detections.py")
        self.command_prefix = self.get_command_prefix(self.args.socket_id) \
            + "{} {}".format(self.python_exe, benchmark_script)
        set_env_var("OMP_NUM_THREADS", self.args.num_intra_threads)

        self.command_prefix += " -g {0}".format(self.args.input_graph)
        self.command_prefix += " -i 1000"
        self.command_prefix += " -w 200"
        self.command_prefix += " -a {0}".format(self.args.num_intra_threads)
        self.command_prefix += " -e {0}".format(self.args.num_inter_threads)
        if self.args.data_location:
            self.command_prefix += " -d {0}".format(self.args.data_location)

        if self.args.accuracy_only:
            self.command_prefix += " -r"
            assert self.args.data_location, "accuracy must provide the data."
        else:
            # Did not support multi-batch accuracy check.
            self.command_prefix += " -b {0}".format(self.args.batch_size)

    def run(self):
        self.run_command(self.command_prefix)
