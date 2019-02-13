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
import sys

from common.base_model_init import BaseModelInitializer
from common.base_model_init import set_env_var


class ModelInitializer(BaseModelInitializer):
    def run_inference_sanity_checks(self, args, custom_args):
        if not args.input_graph:
            sys.exit("Please provide a path to the frozen graph directory"
                     " via the '--in-graph' flag.")
        if not args.data_location:
            sys.exit("Please provide a path to the data directory via the "
                     "'--data-location' flag.")
        if args.socket_id == -1 and args.num_cores == -1:
            print("***Warning***: Running inference on all cores could degrade"
                  " performance. Pass a '--socket-id' to specify running on a"
                  " single socket instead.\n")

    def __init__(self, args, custom_args, platform_util):
        self.args = args
        self.custom_args = custom_args
        self.run_inference_sanity_checks(self.args, self.custom_args)
        self.research_dir = os.path.join(args.model_source_dir, "research")

        # Set KMP env vars, except override the default KMP_BLOCKTIME value
        self.set_kmp_vars(kmp_blocktime="0")

        # set num_inter_threads and num_intra_threads
        self.set_default_inter_intra_threads(platform_util)
        self.args.num_inter_threads = 2
        set_env_var("OMP_NUM_THREADS", self.args.num_intra_threads)

        if self.args.accuracy_only:
            # get accuracy test command
            script_path = os.path.join(
                self.args.benchmark_dir, self.args.use_case,
                self.args.framework, self.args.model_name, self.args.mode,
                "ssdmobilenet_accuracy.sh")
            self.run_cmd = "sh {} {} {}".format(
                script_path, self.args.input_graph, self.args.data_location)
        elif self.args.benchmark_only:
            # get benchmark command
            benchmark_script = os.path.join(
                self.args.benchmark_dir, self.args.use_case,
                self.args.framework, self.args.model_name, self.args.mode,
                self.args.precision, "infer_detections.py")

            # get command with numactl
            self.run_cmd = self.get_numactl_command(
                self.args.socket_id) + "python {}".format(benchmark_script)

            output_tf_record_path = os.path.join(os.path.dirname(
                self.args.data_location), "SSD-mobilenet-out.tfrecord")

            self.run_cmd += " --input_tfrecord_paths={} " \
                            "--output_tfrecord_path={} --inference_graph={} " \
                            "--discard_image_pixels=True " \
                            "--num_inter_threads={} --num_intra_threads={}".\
                format(self.args.data_location, output_tf_record_path,
                       self.args.input_graph, self.args.num_inter_threads,
                       self.args.num_intra_threads)

    def run(self):
        original_dir = os.getcwd()
        os.chdir(self.research_dir)
        self.run_command(self.run_cmd)
        os.chdir(original_dir)
