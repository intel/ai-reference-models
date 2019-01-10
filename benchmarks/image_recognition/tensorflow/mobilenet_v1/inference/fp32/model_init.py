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

import os
from common.base_model_init import BaseModelInitializer

os.environ["KMP_BLOCKTIME"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"


class ModelInitializer(BaseModelInitializer):
    """ Model initializer for MobileNet V1 FP32 inference """

    def __init__(self, args, custom_args=[], platform_util=None):
        self.args = args
        self.custom_args = custom_args
        # use default batch size if -1
        if self.args.batch_size == -1:
            self.args.batch_size = 128

        # set num_inter_threads and num_intra_threads
        self.set_default_inter_intra_threads(platform_util)
        self.args.num_inter_threads = 2

        script_name = "accuracy.py" if self.args.accuracy_only \
            else "eval_image_classifier.py"
        script_path = os.path.join(
            self.args.intelai_models, self.args.mode, self.args.precision,
            script_name)
        self.command_prefix = "python {}".format(script_path)

        if self.args.socket_id != -1:
            self.command_prefix = "numactl --cpunodebind={} -l {}".format(
                str(self.args.socket_id), self.command_prefix)

        os.environ["OMP_NUM_THREADS"] = str(self.args.num_intra_threads)

        if not self.args.accuracy_only:
            self.command_prefix = ("{prefix} "
                                   "--dataset_name imagenet "
                                   "--checkpoint_path {checkpoint} "
                                   "--dataset_dir {dataset} "
                                   "--dataset_split_name=validation "
                                   "--clone_on_cpu=True "
                                   "--model_name {model} "
                                   "--inter_op_parallelism_threads {inter} "
                                   "--intra_op_parallelism_threads {intra} "
                                   "--batch_size {bz}").format(
                prefix=self.command_prefix, checkpoint=self.args.checkpoint,
                dataset=self.args.data_location, model=self.args.model_name,
                inter=self.args.num_inter_threads,
                intra=self.args.num_intra_threads, bz=self.args.batch_size)
        else:
            # add args for the accuracy script
            script_args_list = [
                "input_graph", "data_location", "input_height", "input_width",
                "batch_size", "input_layer", "output_layer",
                "num_inter_threads", "num_intra_threads"]
            self.command_prefix = self.add_args_to_command(
                self.command_prefix, script_args_list)

    def run(self):
        self.run_command(self.command_prefix)
