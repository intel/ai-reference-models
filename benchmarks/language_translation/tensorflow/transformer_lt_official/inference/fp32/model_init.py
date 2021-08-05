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
import sys
from argparse import ArgumentParser

from common.base_model_init import BaseModelInitializer
from common.base_model_init import set_env_var


class ModelInitializer(BaseModelInitializer):
    """Model initializer for Transformer LT FP32 inference"""
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

    def __init__(self, args, custom_args, platform_util=None):
        super(ModelInitializer, self).__init__(args, custom_args, platform_util)

        self.run_inference_sanity_checks(self.args, self.custom_args)
        self.cmd = self.get_command_prefix(self.args.socket_id)
        self.bleu_params = ""

        self.set_num_inter_intra_threads()

        # Set KMP env vars, if they haven't already been set
        config_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        self.set_kmp_vars(config_file_path)

        MODEL_EXEC_DIR = os.path.join(self.args.intelai_models, self.args.mode, self.args.precision)

        set_env_var("OMP_NUM_THREADS", self.args.num_intra_threads)

        if self.args.socket_id != -1:
            if self.args.num_cores != -1:
                self.cmd += "--physcpubind=0-" + \
                            (str(self.args.num_cores - 1)) + " "
        self.cmd += self.python_exe

        run_script = os.path.join(MODEL_EXEC_DIR, "infer_ab.py")

        # Model args
        arg_parser = ArgumentParser(description='process custom_args')
        arg_parser.add_argument('--param_set',
                                help='hparameter setting',
                                dest="param_set",
                                default="big")
        arg_parser.add_argument('--vocab_file',
                                help='input vocable file for translation',
                                dest="vocab_file",
                                default="vocab.txt")
        arg_parser.add_argument('--file',
                                help='decode input file with path',
                                dest="decode_from_file",
                                default="")
        arg_parser.add_argument('--file_out',
                                help='inference output file name',
                                dest="decode_to_file",
                                default="translate.txt")
        arg_parser.add_argument('--reference',
                                help='inference ref file with path',
                                dest="reference",
                                default="")

        self.args = arg_parser.parse_args(self.custom_args,
                                          namespace=self.args)

        # Model parameter control
        translate_file = os.path.join(self.args.output_dir,
                                      self.args.decode_to_file)
        cmd_args = " --param_set=" + self.args.param_set + \
                   " --in_graph=" + self.args.input_graph + \
                   " --batch_size=" + \
                   (str(self.args.batch_size)
                    if self.args.batch_size != -1 else "1") + \
                   " --file=" + self.args.decode_from_file + \
                   " --file_out=" + translate_file + \
                   " --vocab_file=" + self.args.vocab_file +\
                   " --num_inter=" + str(self.args.num_inter_threads) +\
                   " --num_intra=" + str(self.args.num_intra_threads)

        self.bleu_params += " --translation=" + translate_file + \
                            " --reference=" + self.args.reference

        self.cmd += " " + run_script + cmd_args
        compute_bleu_script = os.path.join(MODEL_EXEC_DIR, "compute_bleu.py")
        self.bleucmd = self.python_exe + " " + compute_bleu_script \
            + self.bleu_params

    def run(self):
        original_dir = os.getcwd()
        # os.chdir(self.args.model_source_dir)
        self.run_command(self.cmd)

        # calculate the bleu number after inference is done
        os.system(self.bleucmd)
        os.chdir(original_dir)
