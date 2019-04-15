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
import shutil
from argparse import ArgumentParser

from common.base_model_init import BaseModelInitializer
from common.base_model_init import set_env_var


class ModelInitializer(BaseModelInitializer):
    """Model initializer for Transformer LT FP32 inference"""

    def __init__(self, args, custom_args, platform_util=None):
        super(ModelInitializer, self).__init__(args, custom_args, platform_util)

        self.cmd = self.get_numactl_command(self.args.socket_id)
        self.bleu_params = ""

        self.set_num_inter_intra_threads()

        # Set KMP env vars, if they haven't already been set
        config_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        self.set_kmp_vars(config_file_path)

        TEMP_DIR = str(self.args.model_source_dir) + "/out_dir"
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
        os.mkdir(TEMP_DIR)

        set_env_var("OMP_NUM_THREADS", self.args.num_intra_threads)

        if self.args.socket_id != -1:
            if self.args.num_cores != -1:
                self.cmd += "--physcpubind=0-" + \
                            (str(self.args.num_cores - 1)) + " "
        self.cmd += "{} ".format(self.python_exe)

        run_script = os.path.join(self.args.model_source_dir,
                                  "tensor2tensor/bin/t2t_decoder.py")

        # Model args
        arg_parser = ArgumentParser(description='process custom_args')
        arg_parser.add_argument('-P', '--problem', help='problem name',
                                dest="problem",
                                default="translate_ende_wmt32k")
        arg_parser.add_argument('-M', '--model',
                                help='model name to run', dest="model",
                                default="transformer")
        arg_parser.add_argument('-HPMS', '--hparams_set',
                                help='hparameter setting',
                                dest="hparams_set",
                                default="transformer_base_single_gpu")
        arg_parser.add_argument('-HPM', '--hparams', help='hparameter',
                                dest="hparams", default=None)
        arg_parser.add_argument('-IF', '--decode_from_file',
                                help='decode input file with path',
                                dest="decode_from_file",
                                default=None)
        arg_parser.add_argument('-OF', '--decode_to_file',
                                help='inference output file with path',
                                dest="decode_to_file",
                                default=TEMP_DIR + "/output_infer")
        arg_parser.add_argument('-RF', '--reference',
                                help='inference ref file with path',
                                dest="reference",
                                default=None)

        self.args = arg_parser.parse_args(self.custom_args,
                                          namespace=self.args)

        # Model parameter control
        cmd_args = " --problem=" + self.args.problem + \
                   " --model=" + self.args.model + \
                   " --hparams_set=" + self.args.hparams_set + \
                   " --decode_hparams=beam_size=4,alpha=0.6,batch_size=" + \
                   (str(self.args.batch_size)
                    if self.args.batch_size != -1 else "1") + \
                   " --data_dir=" + str(self.args.data_location) + \
                   " --output_dir=" + self.args.checkpoint + \
                   " --decode_from_file=" + self.args.decode_from_file + \
                   " --decode_to_file=" + self.args.decode_to_file + \
                   " --reference=" + self.args.reference + \
                   " --inter_op_parallelism_threads=" + \
                   str(self.args.num_inter_threads) + \
                   " --intra_op_parallelism_threads=" + \
                   str(self.args.num_intra_threads)

        self.bleu_params += " --translation=" + self.args.decode_to_file + \
                            " --reference=" + self.args.reference

        self.cmd = self.cmd + run_script + cmd_args

    def run(self):
        original_dir = os.getcwd()
        os.chdir(self.args.model_source_dir)
        self.run_command(self.cmd)

        # calculate the bleu number after inference is done
        bleucmd = "python " + \
                  os.path.join(self.args.model_source_dir,
                               "tensor2tensor/bin/t2t_bleu.py") + \
                  self.bleu_params
        os.system(bleucmd)
        os.chdir(original_dir)
