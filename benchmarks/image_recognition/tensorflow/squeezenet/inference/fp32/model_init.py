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

from __future__ import print_function
from common.base_model_init import BaseModelInitializer

import os


class ModelInitializer (BaseModelInitializer):
    """ SqueezeNet model initializer that calls train_squeezenet.py script
     from the models/image_recognition/tensorflow/squeezenet/fp32 directory"""

    def __init__(self, args, custom_args, platform_util):
        super(ModelInitializer, self).__init__(args, custom_args, platform_util)

        cores_per_socket = platform_util.num_cores_per_socket

        # set num_inter_threads and num_intra_threads (override inter threads to 1)
        self.set_num_inter_intra_threads(num_inter_threads=1)

        if self.args.num_cores > 0:
            ncores = self.args.num_cores
        else:
            ncores = self.args.num_intra_threads

        script_path = os.path.join(self.args.intelai_models,
                                   self.args.precision, "train_squeezenet.py")

        self.command = ("taskset -c {:.0f}-{:.0f} python {} "
                        "--data_location {} --batch_size {:.0f} "
                        "--num_inter_threads {:.0f} --num_intra_threads {:.0f}"
                        " --model_dir {} --inference-only").format(
            self.args.socket_id * cores_per_socket,
            ncores - 1 + self.args.socket_id * cores_per_socket,
            script_path, self.args.data_location, self.args.batch_size,
            self.args.num_inter_threads, self.args.num_intra_threads,
            self.args.checkpoint)

        self.command += (' '.join(custom_args)).replace('\t', ' ')

    def run(self):
        if self.args.verbose:
            self.command += " --verbose"

        self.run_command(self.command)
