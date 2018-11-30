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

import argparse
import sys
import os

from common.base_model_init import BaseModelInitializer


class ModelInitializer(BaseModelInitializer):
    args = None
    custom_args = []

    def run_inference_sanity_checks(self, args, custom_args):
        if args.checkpoint is None:
            sys.exit("Please provide a path to the checkpoint"
                     " directory via the '--checkpoint' flag.")
        if args.data_location is None:
            sys.exit("Please provide a path to the data directory "
                     "via the '--data-location' flag.")
        if args.batch_size > 1:
            sys.exit("Currently, only '--batch-size 1' is "
                     "supported during inference.")
        if not args.single_socket and args.num_cores == -1:
            print("***Warning***: Running inference on all cores could "
                  "degrade performance. Pass '--single-socket' instead.\n")

    def __init__(self, args, custom_args, platform_util):
        self.args = args
        self.custom_args = custom_args
        self.parse_args()
        platform_args = platform_util

        if args.mode == "inference":
            self.run_inference_sanity_checks(self.args, self.custom_args)

            if args.single_socket:
                args.num_inter_threads = 1
                if args.num_cores == -1:
                    args.num_intra_threads = \
                        platform_args.num_cores_per_socket()
                else:
                    args.num_intra_threads = args.num_cores

            if args.socket_id == 0:
                cpu_num_begin = 0
            else:
                cpu_num_begin = \
                    args.socket_id * platform_args.num_cores_per_socket()

            if args.num_cores == -1:
                cpu_num_end = (cpu_num_begin + args.num_intra_threads - 1)
            else:
                cpu_num_end = cpu_num_begin + args.num_cores - 1

            self.run_cmd = "numactl --physcpubind=" + str(cpu_num_begin) \
                + "-" + str(cpu_num_end) + " --membind=" \
                + str(args.socket_id) + " taskset -c " \
                + str(cpu_num_begin) + "-" + str(cpu_num_end) \
                + " python -u DeepSpeech.py --log_level 1 " \
                "--checkpoint_dir \"" + str(args.checkpoint) \
                + "\" --one_shot_infer \"" \
                + os.path.join(args.data_location,
                               args.datafile_name) \
                + "\" --inter_op " + str(args.num_inter_threads) \
                + " --intra_op " + str(args.num_intra_threads) \
                + " --num_omp_threads " + str(args.num_intra_threads)

        else:
            # TODO: Add training commands
            sys.exit("Training is currently not supported.")

    def parse_args(self):
        if self.custom_args:
            parser = argparse.ArgumentParser()
            parser.add_argument(
                "--datafile-name", default=None,
                dest='datafile_name', type=str, help="datafile name")

            self.args = parser.parse_args(self.custom_args,
                                          namespace=self.args)

    def run(self):
        os.chdir(os.path.join(self.args.model_source_dir, "DeepSpeech"))
        self.run_command(self.run_cmd)
