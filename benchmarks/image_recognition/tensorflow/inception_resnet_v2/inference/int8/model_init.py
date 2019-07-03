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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from common.base_model_init import BaseModelInitializer, set_env_var

import os


class ModelInitializer(BaseModelInitializer):
    """Model initializer for Inception ResNet V2 int8 inference"""

    def __init__(self, args, custom_args=[], platform_util=None):
        super(ModelInitializer, self).__init__(args, custom_args, platform_util)

        # Set KMP env vars, if they haven't already been set
        config_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        self.set_kmp_vars(config_file_path)

        self.cmd = self.get_command_prefix(self.args.socket_id) + "{} ".format(self.python_exe)

        # use default batch size if -1
        if self.args.batch_size == -1:
            self.args.batch_size = 128

        # set num_inter_threads and num_intra_threads
        self.set_num_inter_intra_threads()

        if self.args.benchmark_only:
            run_script = os.path.join(self.args.intelai_models,
                                      "eval_image_classifier_benchmark.py")

            cmd_args = " --input-graph=" + self.args.input_graph + \
                " --inter-op-parallelism-threads=" + \
                str(self.args.num_inter_threads) + \
                " --intra-op-parallelism-threads=" + \
                str(self.args.num_intra_threads) + \
                " --batch-size=" + str(self.args.batch_size)
        elif self.args.accuracy_only:
            run_script = os.path.join(self.args.intelai_models,
                                      "eval_image_classifier_accuracy.py")

            cmd_args = " --input_graph=" + self.args.input_graph + \
                " --data_location=" + self.args.data_location + \
                " --input_height=299" + " --input_width=299" + \
                " --num_inter_threads=" + str(self.args.num_inter_threads) + \
                " --num_intra_threads=" + str(self.args.num_intra_threads) + \
                " --output_layer=InceptionResnetV2/Logits/Predictions" + \
                " --batch_size=" + str(self.args.batch_size)

        set_env_var("OMP_NUM_THREADS", self.args.num_intra_threads)
        self.cmd = self.cmd + run_script + cmd_args

    def run(self):
        """run command to enable model benchmark or accuracy measurement"""

        if self.cmd:
            self.run_command(self.cmd)
