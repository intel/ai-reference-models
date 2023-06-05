#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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
from argparse import ArgumentParser
import time


class GPTJModelInitializer(BaseModelInitializer):
    """initialize mode and run benchmark"""

    def __init__(self, args, custom_args=[], platform_util=None):
        super(GPTJModelInitializer, self).__init__(args, custom_args, platform_util)

        self.benchmark_command = ""
        if not platform_util:
            raise ValueError("Did not find any platform info.")

        # use default batch size if -1
        if self.args.batch_size == -1:
            self.args.batch_size = 24

        # Set KMP env vars, if they haven't already been set
        config_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        self.set_kmp_vars(config_file_path)

        # set num_inter_threads and num_intra_threads
        self.set_num_inter_intra_threads()

        arg_parser = ArgumentParser(description='Parse args')
        arg_parser.add_argument('--train-option', help=' Training GLUE or SQuAD',
                                dest="train_option", default='GLUE')
        arg_parser.add_argument('--output-dir', help=' Dir to generate outputs', dest="output_dir",
                                default='./output')
        arg_parser.add_argument('--cache-dir', help=' Dir to keep data', dest="cache_dir", default=None)
        arg_parser.add_argument('--do-train', help=' Training ', dest="do_train", default="True")
        arg_parser.add_argument('--init-checkpoint', help=' Checkpoint file', dest="init_checkpoint", default='')
        arg_parser.add_argument('--per-device-train-batch-size', help=' Batch Size',
                                type=int, dest="batch_size", default='24')
        arg_parser.add_argument('--learning-rate', help=' Learning rate', type=float, dest="learning_rate",
                                default=2e-5)
        arg_parser.add_argument('--max-seq-length', help=' max length of sentence to train', dest="max_seq_length",
                                type=int, default=0)
        arg_parser.add_argument('--precision', help='precison fp32, bfloat16 or float16 ',
                                dest="precision", default='bfloat16')
        arg_parser.add_argument('--train-file', help=' Json file for BERT', dest="train_file",
                                default='train-v1.1.json')
        arg_parser.add_argument('--predict-file', help=' Do prediction ', dest="predict_file",
                                default='dev-v1.1.json')
        arg_parser.add_argument('--do-predict', help=' use prediction', dest="do_predict", default="True")
        arg_parser.add_argument('--num-train-epochs', help=' Number of epochs to train ', dest="num_train_epochs",
                                type=float, default=2)
        arg_parser.add_argument('--do-eval', help=' Eval for pretraing', dest="do_eval", default="True")
        arg_parser.add_argument('--num-train-steps', help=' Number of steps to train ', dest="num_train_steps",
                                type=int, default=20)
        arg_parser.add_argument('--warmup-steps', help=' Number of warmup steps', dest="warmup_steps", type=int,
                                default=10)
        arg_parser.add_argument('--max-predictions', help=' max predictions:pretraining', dest="max_predictions",
                                type=int, default=20)
        arg_parser.add_argument('--task-name', help=' Task name for classifier', dest="task_name", default='MRPC')
        arg_parser.add_argument('--num-inter-threads', help=' Number of Inter ops threads', type=int,
                                dest="num_inter_threads", default=self.args.num_inter_threads)
        arg_parser.add_argument('--num-intra-threads', help=' Number of Intra ops threads', type=int,
                                dest="num_intra_threads", default=self.args.num_inter_threads)
        arg_parser.add_argument('--profile', help=' Enable Tensorflow profiler hook',
                                dest="profile", default="False")

        self.args = arg_parser.parse_args(self.custom_args, namespace=self.args)

        # Set KMP env vars, if they haven't already been set, but override the default KMP_BLOCKTIME value
        self.set_kmp_vars(config_file_path)

        set_env_var("OMP_NUM_THREADS", self.args.num_intra_threads)

        run_script = "run_glue.py"
        if self.args.train_option == "Lambada":
            run_script = "run_lambada.py"
        elif self.args.train_option == "GLUE":
            run_script = "run_glue.py"
        else:
            print("INFO:Running GLUE {}...!".format(self.args.task_name))

        benchmark_script = os.path.join(
            self.args.intelai_models, self.args.mode,
            run_script)

        if os.environ["MPI_NUM_PROCESSES"] == "None":
            self.benchmark_command = self.get_command_prefix(args.socket_id)
        else:
            if os.environ["MPI_NUM_PROCESSES_PER_SOCKET"] == "1":
                # Map by socket using OpenMPI by default (PPS=1).
                self.benchmark_command = "mpirun --allow-run-as-root --map-by socket "

        eoo = " \\\n"
        self.cmd_args = \
            " --do_train=" + str(self.args.do_train) + eoo + \
            " --per_device_train_batch_size=" + str(self.args.batch_size) + eoo + \
            " --learning_rate=" + str(self.args.learning_rate) + eoo + \
            " --intra_op_parallelism_threads=" + str(self.args.num_intra_threads) + eoo + \
            " --inter_op_parallelism_threads=" + str(self.args.num_inter_threads) + eoo + \
            " --output_dir=" + str(self.args.output_dir) + eoo + \
            " --overwrite_output_dir " + eoo + \
            " --profile=" + str(self.args.profile) + eoo
        self.cmd_args += " --block_size=" + str(self.args.max_seq_length) + eoo if self.args.max_seq_length else ""
        self.cmd_args += " --precision=" + self.args.precision + eoo
        self.cmd_args += " --num_train_epochs=" + str(self.args.num_train_epochs) + eoo
        if self.args.cache_dir:
            self.cmd_args += " --cache_dir=" + str(self.args.cache_dir) + eoo

        if self.args.train_option == "SQuAD":
            self.cmd_args = self.cmd_args + \
                " --vocab_file=" + str(self.args.vocab_file) + eoo + \
                " --train_file=" + str(self.args.train_file) + eoo + \
                " --predict_file=" + str(self.args.predict_file) + eoo + \
                " --do_predict=" + str(self.args.do_predict) + eoo + \
                " --model_name_or_path=" + str(self.args.init_checkpoint)

        if self.args.train_option == "GLUE":
            self.cmd_args = self.cmd_args + \
                " --task_name=" + str(self.args.task_name) + eoo + \
                " --do_eval=" + str(self.args.do_eval) + eoo + \
                " --do_predict=" + str(self.args.do_predict) + eoo

        if (self.args.init_checkpoint and len(self.args.init_checkpoint) > 0):
            self.cmd_args += eoo + " --model_name_or_path=" + str(self.args.init_checkpoint)

        benchmark_script = benchmark_script + eoo + self.cmd_args

        if self.args.data_num_inter_threads:
            benchmark_script += " --data-num-inter-threads=" + str(self.args.data_num_inter_threads)
        if self.args.data_num_intra_threads:
            benchmark_script += " --data-num-intra-threads=" + str(self.args.data_num_intra_threads)

        if os.environ["MPI_NUM_PROCESSES"] == "None":
            self.benchmark_command = self.benchmark_command + self.python_exe + " " + benchmark_script + "\n"
        else:
            numa_cmd = " -np 1 numactl -N {} -m {} "
            self.benchmark_command = self.benchmark_command + numa_cmd.format(0, 0) + os.environ["PYTHON_EXE"] + " " \
                + benchmark_script
            for i in range(1, int(os.environ["MPI_NUM_PROCESSES"])):
                self.benchmark_command = self.benchmark_command + eoo + " : " + numa_cmd.format(i, i) \
                    + os.environ["PYTHON_EXE"] + " " + benchmark_script

        # if output results is enabled, generate a results file name and pass it to the script
        if self.args.output_results:
            self.results_filename = "{}_{}_{}_results_{}.txt".format(
                self.args.model_name, self.args.precision, self.args.mode,
                time.strftime("%Y%m%d_%H%M%S", time.gmtime()))
            self.results_file_path = os.path.join(self.args.output_dir, self.results_filename)
            self.benchmark_command += " --results-file-path {}".format(self.results_file_path)

    def run(self):
        if self.benchmark_command:
            print("----------------------------Run command-------------------------------------")
            print(self.benchmark_command)
            print("------------------------------------------------------------------------")
            self.run_command(self.benchmark_command)
            if self.args.output_results:
                print("Training results file in the output directory: {}".format(self.results_filename))
