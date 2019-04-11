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
# SPDX-License-Identifier: EPL-2.0
#

import os
from argparse import ArgumentParser

from common.base_model_init import BaseModelInitializer
from common.base_model_init import set_env_var


class ModelInitializer(BaseModelInitializer):
    """Model initializer for GNMT FP32 inference"""

    def __init__(self, args, custom_args=[], platform_util=None):
        super(ModelInitializer, self).__init__(args, custom_args, platform_util)
        self.cmd = self.get_numactl_command(self.args.socket_id)

        if self.args.socket_id != -1 and self.args.num_cores != -1:
            self.cmd += "--physcpubind=0-" + \
                        (str(self.args.num_cores - 1)) + " "
        self.cmd += "{} ".format(self.python_exe)

        # Set the KMP env vars
        self.set_kmp_vars(kmp_affinity="granularity=fine,compact,1,0")

        # use default batch size if -1
        if self.args.batch_size == -1:
            self.args.batch_size = 32

        # set num_inter_threads and num_intra_threads (override inter threads to 2)
        self.set_num_inter_intra_threads()

        DATA_DIR = os.path.join(self.args.intelai_models,
                                self.args.precision, "wmt16")
        HPARAM_DIR = os.path.join(self.args.intelai_models,
                                  self.args.precision, "standard_hparams")

        set_env_var("OMP_NUM_THREADS", self.args.num_intra_threads)

        if self.args.data_location is None:
            data_dir = DATA_DIR
        else:
            data_dir = self.args.data_location

        arg_parser = ArgumentParser(description="process custom_args")
        arg_parser.add_argument(
            "-S", "--src", help="source lanuage",
            dest="src", default="de")
        arg_parser.add_argument(
            "-T", "--tgt", help="target lanuage",
            dest="tgt", default="en")
        arg_parser.add_argument(
            "-HPMS", "--hparams_path", help="hparameter files location",
            dest="hparams_path",
            default=HPARAM_DIR + "/wmt16_gnmt_4_layer_internal.json")
        arg_parser.add_argument(
            "-VP", "--vocab_prefix",
            help="prefix of vocabulary file",
            dest="vocab_prefix",
            default=data_dir + "/vocab.bpe.32000")
        arg_parser.add_argument(
            "-IF", "--inference_input_file",
            help="inference input file with path",
            dest="inference_input_file",
            default=data_dir + "/newstest2015.tok.bpe.32000.de")
        arg_parser.add_argument(
            "-OF", "--inference_output_file",
            help="inference output file with path",
            dest="inference_output_file",
            default=self.args.output_dir + "/output_infer")
        arg_parser.add_argument(
            "-RF", "--inference_ref_file",
            help="inference ref file with path",
            dest="inference_ref_file",
            default=data_dir + "/newstest2015.tok.bpe.32000.en")
        arg_parser.add_argument(
            "--infer_mode", type=str, default="greedy",
            choices=["greedy", "sample", "beam_search"],
            help="Which type of decoder to use during inference.",
            dest="infer_mode")

        self.args = arg_parser.parse_args(self.custom_args, namespace=self.args)

        # Model parameter control

        cmd_args = " --src=" + self.args.src + " --tgt=" + self.args.tgt + \
                   " --hparams_path=" + self.args.hparams_path + \
                   " --out_dir=" + self.args.output_dir + \
                   " --vocab_prefix=" + self.args.vocab_prefix + \
                   " --ckpt=" + (self.args.checkpoint + "/translate.ckpt") + \
                   " --infer_batch_size=" + str(self.args.batch_size) + \
                   " --inference_input_file=" + \
                   self.args.inference_input_file + \
                   " --inference_output_file=" + \
                   self.args.inference_output_file + \
                   " --inference_ref_file=" + self.args.inference_ref_file + \
                   " --num_inter_threads=" + \
                   str(self.args.num_inter_threads) + \
                   " --num_intra_threads=" + \
                   str(self.args.num_intra_threads) + \
                   " --infer_mode=" + self.args.infer_mode

        run_script = os.path.join(self.args.intelai_models,
                                  self.args.precision, "nmt.py")

        self.cmd = self.cmd + run_script + cmd_args

    def run(self):
        if self.cmd:
            self.run_command(self.cmd)
