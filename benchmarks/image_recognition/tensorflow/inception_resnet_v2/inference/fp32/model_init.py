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

from common.base_model_init import BaseModelInitializer
from common.base_model_init import set_env_var

import os


class ModelInitializer(BaseModelInitializer):
    """Model initializer for Inception Resnet V2 FP32 inference"""

    def __init__(self, args, custom_args=[], platform_util=None):
        super(ModelInitializer, self).__init__(args, custom_args, platform_util)
        self.cmd = self.get_numactl_command(self.args.socket_id) + self.python_exe + " "

        # Set KMP env vars, if they haven't already been set
        self.set_kmp_vars()

        # use default batch size if -1
        if self.args.batch_size == -1:
            self.args.batch_size = 128

        # set num_inter_threads and num_intra_threads (override inter threads to 2)
        self.set_num_inter_intra_threads(num_inter_threads=2)

        set_env_var("OMP_NUM_THREADS", self.args.num_intra_threads)

        if self.args.benchmark_only:
            run_script = os.path.join(self.args.intelai_models,
                                      "eval_image_classifier.py")

            cmd_args = " --dataset_name=imagenet" + \
                " --checkpoint_path=" + self.args.checkpoint + \
                " --eval_dir=" + self.args.checkpoint + \
                " --dataset_dir=" + self.args.data_location + \
                " --dataset_split_name=validation" + \
                " --clone_on_cpu=True" + \
                " --model_name=" + str(self.args.model_name) + \
                " --inter_op_parallelism_threads=" + \
                str(self.args.num_inter_threads) + \
                " --intra_op_parallelism_threads=" + \
                str(self.args.num_intra_threads) + \
                " --batch_size=" + str(self.args.batch_size)
        elif self.args.accuracy_only:
            run_script = os.path.join(self.args.intelai_models,
                                      "eval_image_classifier_accuracy.py")
            cmd_args = " --input_graph=" + self.args.input_graph + \
                " --data_location=" + self.args.data_location + \
                " --input_height=299" + " --input_width=299" + \
                " --num_inter_threads=" + \
                       str(self.args.num_inter_threads) + \
                " --num_intra_threads=" + \
                       str(self.args.num_intra_threads) + \
                " --output_layer=InceptionResnetV2/Logits/Predictions" + \
                " --batch_size=" + str(self.args.batch_size)

        self.cmd = self.cmd + run_script + cmd_args

    def run(self):
        """run command to enable model benchmark or accuracy measurement"""

        if self.cmd:
            self.run_command(self.cmd)
