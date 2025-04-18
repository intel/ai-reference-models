#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
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
from common.base_model_init import BaseModelInitializer
from common.base_model_init import set_env_var

import os
import sys
from argparse import ArgumentParser


class ModelInitializer(BaseModelInitializer):
    """initialize mode and run benchmark"""

    def __init__(self, args, custom_args=[], platform_util=None):
        super(ModelInitializer, self).__init__(args, custom_args, platform_util)

        self.benchmark_command = ""
        if not platform_util:
            raise ValueError("Did not find any platform info.")

        # multi-instance runs on bert require each instance to have separate output folders
        # and the output folders should be empty to prevent conflicts with previous runs
        if self.args.numa_cores_per_instance and os.path.exists(self.args.output_dir):
            if os.listdir(self.args.output_dir):
                sys.exit("ERROR: The output directory ({}) is not empty. BERT multi-instance "
                         "runs require empty output directories in order to prevent conflicts "
                         "with files generated by previous runs. Please provide an empty folder "
                         "using the --output-dir argument.".format(self.args.output_dir))

        # BERT Int8 inference requires a frozen graph
        if not self.args.input_graph:
            sys.exit(
                "ERROR: BERT large Int8 inference requires a frozen graph to be passed "
                "to the launch_benchmark.py script using the --in-graph arg. Please download "
                "the frozen graph using the instructions in the README.md and update your command "
                "to add the input graph.")

        # use default batch size of 32 if it's -1
        if self.args.batch_size == -1:
            self.args.batch_size = 32

        arg_parser = ArgumentParser(description="Parse bert inference args")
        arg_parser.add_argument('--infer-option', help=' Inference SQuAD, Pretraining or classifier',
                                dest="infer_option", default='SQuAD')
        arg_parser.add_argument(
            "--doc-stride", dest="doc_stride", type=int, default=None)
        arg_parser.add_argument(
            "--max-seq-length", type=int, dest="max_seq_length", default=None)
        arg_parser.add_argument(
            "--profile", dest="profile", default=None)
        arg_parser.add_argument(
            "--config-file", dest="bert_config_file", default="bert_config.json")
        arg_parser.add_argument(
            "--vocab-file", dest="vocab_file", default="vocab.txt")
        arg_parser.add_argument(
            "--predict-file", dest="predict_file", default="dev-v1.1.json")
        arg_parser.add_argument(
            "--init-checkpoint", dest="init_checkpoint", default="model.ckpt-3649")
        arg_parser.add_argument(
            "--experimental-gelu", dest="experimental_gelu", default="False")
        arg_parser.add_argument(
            "--optimized-softmax", dest="optimized_softmax", default="True")
        arg_parser.add_argument("--warmup-steps", dest='warmup_steps',
                                type=int, default=10,
                                help="number of warmup steps")
        arg_parser.add_argument("--steps", dest='steps',
                                type=int, default=30,
                                help="number of benchmark steps")
        arg_parser.add_argument("--weight-sharing", dest="weight_sharing", action="store_false")
        self.args = arg_parser.parse_args(self.custom_args, namespace=self.args)

        # Set KMP env vars, if they haven't already been set, but override the default KMP_BLOCKTIME value
        config_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        self.set_kmp_vars(config_file_path)

        # Get full paths to files. Assume that they are in the data_location,
        # unless we are given the full path to the file (which might happen for bare metal)
        if self.args.vocab_file and not os.path.isabs(self.args.vocab_file):
            self.args.vocab_file = os.path.join(self.args.data_location, self.args.vocab_file)

        if self.args.bert_config_file and not os.path.isabs(self.args.bert_config_file):
            self.args.bert_config_file = os.path.join(self.args.data_location, self.args.bert_config_file)

        if self.args.predict_file and not os.path.isabs(self.args.predict_file):
            self.args.predict_file = os.path.join(self.args.data_location, self.args.predict_file)

        if self.args.init_checkpoint and not os.path.isabs(self.args.init_checkpoint):
            self.args.init_checkpoint = os.path.join(self.args.checkpoint, self.args.init_checkpoint)

        # set default inter/intra threads
        self.set_num_inter_intra_threads()

        if not os.getenv("OMP_NUM_THREADS"):
            if self.args.num_intra_threads:
                set_env_var("OMP_NUM_THREADS", self.args.num_intra_threads)
            else:
                set_env_var("OMP_NUM_THREADS", platform_util.num_cores_per_socket)

        if self.args.onednn_graph:
            model_script = os.path.join(
                self.args.intelai_models, self.args.mode, "run_squad_int8.py")
        else:
            model_script = os.path.join(
                self.args.intelai_models, self.args.mode, "run_squad.py")

        model_args = " --init_checkpoint=" + str(self.args.init_checkpoint) + \
                     " --vocab_file=" + str(self.args.vocab_file) + \
                     " --bert_config_file=" + str(self.args.bert_config_file) + \
                     " --predict_file=" + str(self.args.predict_file) + \
                     " --precision=" + str(self.args.precision) + \
                     " --output_dir=" + str(self.args.output_dir) + \
                     " --predict_batch_size=" + str(self.args.batch_size) + \
                     " --experimental_gelu=" + str(self.args.experimental_gelu) + \
                     " --optimized_softmax=" + str(self.args.optimized_softmax) + \
                     " --input_graph=" + str(self.args.input_graph) + \
                     " --do_predict=True "

        if self.args.accuracy_only:
            model_args += " --mode=accuracy"

        if self.args.profile and self.args.profile.lower() == "true":
            model_args += " --mode=profile"

        if self.args.benchmark_only:
            model_args += " --mode=benchmark"

        if self.args.doc_stride:
            model_args += " --doc_stride=" + str(self.args.doc_stride)

        if self.args.max_seq_length:
            model_args += " --max_seq_length=" + str(self.args.max_seq_length)

        if self.args.num_inter_threads:
            model_args += " --inter_op_parallelism_threads=" + str(self.args.num_inter_threads)

        if self.args.num_intra_threads:
            model_args += " --intra_op_parallelism_threads=" + str(self.args.num_intra_threads)

        if self.args.warmup_steps:
            model_args += " --warmup_steps=" + str(self.args.warmup_steps)

        if self.args.steps:
            model_args += " --steps=" + str(self.args.steps)

        if self.args.weight_sharing:
            model_args += " --weight_sharing"

        model_args += " --num_cores_per_socket=" + str(platform_util.num_cores_per_socket)

        self.benchmark_command = self.get_command_prefix(args.socket_id) + \
            self.python_exe + " " + model_script + model_args

    def run(self):
        if self.benchmark_command:
            # Bert needs to have unique output directories for every instance, so pass the output_dir
            # string to the run_command function so that unique dirs get swapped in to the commands
            self.run_command(self.benchmark_command,
                             replace_unique_output_dir=str(self.args.output_dir))

        # execute the evaluate script for accuracy mode
        if self.args.accuracy_only:
            evaluate_script = os.path.join(
                self.args.intelai_models, self.args.mode, "evaluate-v1.1.py")
            predictions_json = os.path.join(self.args.output_dir, "predictions.json")
            if os.path.exists(predictions_json):
                evaluate_cmd = "{} {} {} {}".format(
                    self.python_exe, evaluate_script, self.args.predict_file,
                    predictions_json)
                os.system(evaluate_cmd)
            else:
                print("Warning: The {} preditions file was not found. Unable to "
                      "run the evaluation script.".format(predictions_json))
