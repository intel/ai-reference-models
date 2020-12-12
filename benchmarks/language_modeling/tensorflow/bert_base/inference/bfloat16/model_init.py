#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Intel Corporation
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

import os
from argparse import ArgumentParser


class ModelInitializer(BaseModelInitializer):
    """initialize mode and run benchmark"""

    def __init__(self, args, custom_args=[], platform_util=None):
        super(ModelInitializer, self).__init__(args, custom_args, platform_util)

        self.benchmark_command = ""
        if not platform_util:
            raise ValueError("Did not find any platform info.")

        # use default batch size of 32 if it's -1
        if self.args.batch_size == -1:
            self.args.batch_size = 32

        self.set_num_inter_intra_threads()

        config_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        self.set_kmp_vars(config_file_path)

        arg_parser = ArgumentParser(description="Parse bert inference args")
        arg_parser.add_argument('--infer-option', help=' Inference Classifier', dest="infer_option",
                                default='Classifier')
        arg_parser.add_argument("--max-seq-length", type=int, dest="max_seq_length", default=None)
        arg_parser.add_argument("--profile", dest="profile", default=None)
        arg_parser.add_argument('--experimental-gelu', help=' [Experimental] Use experimental gelu op.',
                                dest="experimental_gelu", default="False")
        arg_parser.add_argument("--config-file", dest="bert_config_file", default="bert_config.json")
        arg_parser.add_argument("--vocab-file", dest="vocab_file", default="vocab.txt")
        arg_parser.add_argument('--task-name', help=' Task name for classifier', dest="task_name", default='MRPC')
        arg_parser.add_argument('--do-eval', help=' Eval for Classifier', dest="do_eval",
                                default="False")  # compatible with SQuAD
        arg_parser.add_argument('--data-dir', help=' data dir for Classifier', dest="data_dir", default='MRPC')
        arg_parser.add_argument('--do-lower-case', help=' Use lowercase for data',
                                dest="do_lower_case", default="False")  # compatible with training

        self.args = arg_parser.parse_args(self.custom_args, namespace=self.args)

        # Get full paths to files. Assume that they are in the data_location,
        # unless we are given the full path to the file (which might happen for bare metal)
        def expand_data_path(path):
            if path and not os.path.isabs(path):
                path = os.path.join(self.args.data_location, path)

            return path

        self.args.vocab_file = expand_data_path(self.args.vocab_file)
        self.args.bert_config_file = expand_data_path(self.args.bert_config_file)

        run_script = ""
        if self.args.infer_option == "Classifier":
            run_script = "run_classifier.py"
            print("INFO:Running Classify...!")
        else:
            print("ERROR: only support classifier now")

        # we reuse the same code base with bert large.
        model_script = os.path.join(self.args.intelai_models, '../bert_large', self.args.mode, run_script)

        print(model_script)

        model_args = ""

        # if defined input_graph (frozen graph), use it. otherwise, use the checkpoint in output dir.
        if self.args.input_graph:
            model_args = " --frozen_graph_path=" + str(self.args.input_graph)

        eoo = " \\\n"
        model_args = model_args + \
            " --output_dir=" + str(self.args.output_dir) + eoo + \
            " --bert_config_file=" + str(self.args.bert_config_file) + eoo + \
            " --do_train=" + str(False) + eoo + \
            " --precision=" + str(self.args.precision) + eoo + \
            " --do_lower_case=" + str(self.args.do_lower_case)

        if self.args.infer_option == "SQuAD":
            model_args = model_args + \
                " --vocab_file=" + str(self.args.vocab_file) + eoo + \
                " --predict_file=" + str(self.args.predict_file) + eoo + \
                " --do_predict=True"

        if self.args.infer_option == "Classifier":
            model_args = model_args + \
                " --task_name=" + str(self.args.task_name) + eoo + \
                " --do_eval=" + str(self.args.do_eval) + eoo + \
                " --vocab_file=" + str(self.args.vocab_file) + eoo + \
                " --data_dir=" + str(self.args.data_dir) + eoo + \
                " --eval_batch_size=" + str(self.args.batch_size) + \
                " --experimental_gelu=" + str(self.args.experimental_gelu)

        if self.args.accuracy_only:
            model_args += " --mode=accuracy"

        if self.args.profile and self.args.profile.lower() == "true":
            model_args += " --mode=profile"
            model_args += " --profile=True"

        if self.args.benchmark_only:
            model_args += " --mode=benchmark"

        if self.args.max_seq_length:
            model_args += " --max_seq_length=" + str(self.args.max_seq_length)

        if self.args.num_inter_threads:
            model_args += " --inter_op_parallelism_threads=" + str(self.args.num_inter_threads)

        if self.args.num_intra_threads:
            model_args += " --intra_op_parallelism_threads=" + str(self.args.num_intra_threads)

        self.benchmark_command = self.get_command_prefix(args.socket_id) + \
            self.python_exe + " " + model_script + model_args

    def run(self):
        if self.benchmark_command:
            self.run_command(self.benchmark_command)
