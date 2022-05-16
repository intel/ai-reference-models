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
#

import os
from argparse import ArgumentParser

from common.base_model_init import BaseModelInitializer


class ModelInitializer(BaseModelInitializer):
    """Model initializer for Transformer LT int8 inference"""

    def __init__(self, args, custom_args, platform_util=None):
        super(ModelInitializer, self).__init__(args, custom_args, platform_util)

        self.cmd = self.get_command_prefix(self.args.socket_id)
        self.bleu_params = ""

        MODEL_EXEC_DIR = os.path.join(self.args.intelai_models, self.args.mode, self.args.precision)

        self.cmd += self.python_exe

        run_script = os.path.join(MODEL_EXEC_DIR, "transformer/translate.py")

        # Model args
        arg_parser = ArgumentParser(description='process custom_args')
        arg_parser.add_argument('--params',
                                help='transformer model setting',
                                dest="params",
                                default="big")
        arg_parser.add_argument('--vocab_file',
                                help='input vocab file for inference directory',
                                dest="vocab_file",
                                default="")
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
        arg_parser.add_argument("--warmup-steps", dest='warmup_steps',
                                type=int, default=3,
                                help="number of warmup steps")
        arg_parser.add_argument("--steps", dest='steps',
                                type=int, default=100,
                                help="number of steps")
        arg_parser.add_argument('--kmp-blocktime', dest='kmp_blocktime',
                                help='number of kmp block time',
                                type=int, default=1)

        self.args = arg_parser.parse_args(self.custom_args,
                                          namespace=self.args)

        # Set KMP env vars, if they haven't already been set
        config_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        self.set_kmp_vars(config_file_path, kmp_blocktime=str(self.args.kmp_blocktime))

        # Model parameter control
        translate_file = os.path.join(self.args.output_dir,
                                      self.args.decode_to_file)
        if self.args.benchmark_only:
            testmode = 'benchmark'
        elif self.args.accuracy_only:
            testmode = 'accuracy'
        else:
            testmode = self.args.mode
        cmd_args = " --params=" + self.args.params + \
                   " --input_graph=" + self.args.input_graph + \
                   " --batch_size=" + \
                   (str(self.args.batch_size)
                    if self.args.batch_size != -1 else "1") + \
                   " --test_mode=" + testmode + \
                   " --warmup_steps=" + str(self.args.warmup_steps) + \
                   " --steps=" + str(self.args.steps) + \
                   " --vocab_file=" + self.args.vocab_file + \
                   " --file=" + self.args.decode_from_file + \
                   " --file_out=" + translate_file + \
                   " --data_dir=" + self.args.data_location + \
                   " --num_inter=" + str(self.args.num_inter_threads) + \
                   " --num_intra=" + str(self.args.num_intra_threads)

        self.bleu_params += " --translation=" + translate_file + \
                            " --reference=" + self.args.reference

        self.cmd += " " + run_script + cmd_args
        compute_bleu_script = os.path.join(MODEL_EXEC_DIR, "transformer/compute_bleu.py")
        if self.args.accuracy_only:
            self.bleucmd = self.python_exe + " " + compute_bleu_script + self.bleu_params
        else:
            self.bleucmd = ''

    def run(self):
        # TODO: make it a property in PlatformUtils (platform_util.os_type) to get the host OS.
        # We already do the OS check there to see if it's one that we support.
        if os.environ.get('OS', '') == 'Windows_NT':
            os.environ["PYTHONPATH"] = "{};{}".format(
                os.path.join(self.args.intelai_models, os.pardir,
                             os.pardir, os.pardir, "common", "tensorflow"),
                os.environ["PYTHONPATH"])
        original_dir = os.getcwd()
        print(self.cmd)
        self.run_command(self.cmd)

        # calculate the bleu number after inference is done
        os.system(self.bleucmd)
        os.chdir(original_dir)
