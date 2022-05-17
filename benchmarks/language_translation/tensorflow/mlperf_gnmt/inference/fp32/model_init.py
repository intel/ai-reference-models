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
from argparse import ArgumentParser

from common.base_model_init import BaseModelInitializer
from common.base_model_init import set_env_var


class ModelInitializer(BaseModelInitializer):
    """Model initializer for GNMT FP32 inference"""

    def __init__(self, args, custom_args=[], platform_util=None):
        super(ModelInitializer, self).__init__(args, custom_args, platform_util)
        self.cmd = self.get_command_prefix(self.args.socket_id)

        if self.args.socket_id != -1 and self.args.num_cores != -1:
            self.cmd += "--physcpubind=0-" + \
                        (str(self.args.num_cores - 1)) + " "
        self.cmd += "{} ".format(self.python_exe)

        # use default batch size if -1
        if self.args.batch_size == -1:
            self.args.batch_size = 32

        # set num_inter_threads and num_intra_threads (override inter threads to 2)
        self.set_num_inter_intra_threads()
        set_env_var("OMP_NUM_THREADS", self.args.num_intra_threads)
        arg_parser = ArgumentParser(description="process custom_args")
        arg_parser.add_argument('--kmp-blocktime', dest='kmp_blocktime',
                                help='number of kmp block time',
                                type=int, default=1)
        self.args = arg_parser.parse_args(self.custom_args, namespace=self.args)
        # Set KMP env vars, if they haven't already been set, but override the default KMP_BLOCKTIME value
        config_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        self.set_kmp_vars(config_file_path, kmp_blocktime=str(self.args.kmp_blocktime))

        src_vocab_file = os.path.join(self.args.data_location, "vocab.bpe.32000.en")
        tgt_vocab_file = os.path.join(self.args.data_location, "vocab.bpe.32000.de")
        inference_input_file = os.path.join(self.args.data_location, "newstest2014.tok.bpe.32000.en")
        inference_ref_file = os.path.join(self.args.data_location, "newstest2014.tok.bpe.32000.de")

        cmd_args = " --in_graph=" + self.args.input_graph + \
                   " --batch_size=" + str(self.args.batch_size) + \
                   " --num_inter_threads=" + str(self.args.num_inter_threads) + \
                   " --num_intra_threads=" + str(self.args.num_intra_threads) + \
                   " --src_vocab_file=" + src_vocab_file + \
                   " --tgt_vocab_file=" + tgt_vocab_file + \
                   " --inference_input_file=" + inference_input_file + \
                   " --inference_ref_file=" + inference_ref_file

        run_script = os.path.join(self.args.intelai_models,
                                  self.args.precision, "run_inference.py")

        self.cmd = self.cmd + run_script + cmd_args

    def run(self):
        if self.cmd:
            self.run_command(self.cmd)
