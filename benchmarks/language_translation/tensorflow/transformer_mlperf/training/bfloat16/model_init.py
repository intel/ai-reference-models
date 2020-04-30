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

DEFAULT_TRAIN_EPOCHS = 10
BLEU_DIR = "bleu"
INF = 10000

class ModelInitializer(BaseModelInitializer):
    """Model initializer for Transformer LT FP32 inference"""

    def __init__(self, args, custom_args, platform_util=None):
        super(ModelInitializer, self).__init__(args, custom_args, platform_util)

        self.cmd = self.get_command_prefix(self.args.socket_id)
        self.bleu_params = ""

        self.set_num_inter_intra_threads()

        # Set KMP env vars, if they haven't already been set
        config_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        self.set_kmp_vars(config_file_path)

        set_env_var("OMP_NUM_THREADS", self.args.num_intra_threads)

        if self.args.socket_id != -1:
            if self.args.num_cores != -1:
                self.cmd += "--physcpubind=0-" + \
                            (str(self.args.num_cores - 1)) + " "
        self.cmd += "{} ".format(self.python_exe)

        #run_script = os.path.join(self.args.model_source_dir,
        #                          "tensor2tensor/bin/t2t_decoder.py")
        run_script = os.path.join(
                            self.args.intelai_models, self.args.mode,
                                        "bfloat16/transformer/transformer_main.py")

        parser = ArgumentParser(description='process custom_args')
        # Flags for training with epochs. (default)
        parser.add_argument(
            "--train_epochs", "-te", type=int, default=None,
            help="The number of epochs used to train. If both --train_epochs and "
                 "--train_steps are not set, the model will train for %d epochs." %
            DEFAULT_TRAIN_EPOCHS,
            metavar="<TE>")
        parser.add_argument(
            "--epochs_between_eval", "-ebe", type=int, default=1,
            help="[default: %(default)s] The number of training epochs to run "
                 "between evaluations.",
            metavar="<TE>")

        # Flags for training with steps (may be used for debugging)
        parser.add_argument(
            "--train_steps", "-ts", type=int, default=0,
            help="Total number of training steps. If both --train_epochs and "
                 "--train_steps are not set, the model will train for %d epochs." %
            DEFAULT_TRAIN_EPOCHS,
            metavar="<TS>")
        parser.add_argument(
            "--steps_between_eval", "-sbe", type=int, default=1000,
            help="[default: %(default)s] Number of training steps to run between "
                 "evaluations.",
            metavar="<SBE>")

        # BLEU score computation
        parser.add_argument(
            "--bleu_source", "-bs", type=str, default=None,
            help="Path to source file containing text translate when calculating the "
                 "official BLEU score. Both --bleu_source and --bleu_ref must be "
                 "set. The BLEU score will be calculated during model evaluation.",
            metavar="<BS>")
        parser.add_argument(
            "--bleu_ref", "-br", type=str, default=None,
            help="Path to file containing the reference translation for calculating "
                 "the official BLEU score. Both --bleu_source and --bleu_ref must be "
                 "set. The BLEU score will be calculated during model evaluation.",
            metavar="<BR>")
        parser.add_argument(
            "--bleu_threshold", "-bt", type=float, default=None,
            help="Stop training when the uncased BLEU score reaches this value. "
                 "Setting this overrides the total number of steps or epochs set by "
                 "--train_steps or --train_epochs.",
            metavar="<BT>")
        parser.add_argument(
          "--random_seed", "-rs", type=int, default=None,
          help="the random seed to use", metavar="<SEED>")
        parser.add_argument(
      	  "--params", "-p", type=str, default="big", choices=["base", "big"],
      	  help="[default: %(default)s] Parameter set to use when creating and "
          "training the model.",
          metavar="<P>") 
        parser.add_argument(
      	  "--do_eval", "-de", type=str, default="No", choices=["Yes", "No"],
      	  help="[default: %(default)s] set, to not do  evaluation "
          "to reduce train time.",
          metavar="<DE>") 
        parser.add_argument(
      	  "--save_checkpoints", "-sc", type=str, default="No", choices=["Yes", "No"],
      	  help="[default: %(default)s]  set, to not saving checkpoints "
          "to reduce training time.",
          metavar="<SC>") 
        parser.add_argument(
      	  "--save_profile", "-sp", type=str, default="No", 
      	  help="[default: %(default)s]  set, to not saving profiles "
          "to reduce training time.",
          metavar="<SP>") 
        parser.add_argument(
      	  "--print_iter", "-pi", type=int, default="10",
      	  help="[default: %(default)s]  set, to print in every 10 iterations "
          "to reduce print time",
          metavar="<PI>") 
        parser.add_argument(
      	  "--learning_rate", "-lr", type=int, default="2",
      	  help="[default: %(default)s]  set learning rate 2 "
          "or can be set",
          metavar="<LR>") 
        parser.add_argument(
      	  "--static_batch", "-sb", type=str, default="No",
      	  help="[default: %(default)s]  set, to not using static batch ",
          metavar="<SB>")

        #Ashraf: work with the platform.py file to add the following arg
        parser.add_argument(
          "--num_cpu_cores", "-nc", type=int, default=4,
          help="[default: %(default)s] Number of CPU cores to use in the input "
               "pipeline.",
          metavar="<NC>")

        self.args = parser.parse_args(self.custom_args,
                                          namespace=self.args)
        # Model parameter control 
        #TODO: need more arguments for full training
        cmd_args = " --data_dir=" + str(self.args.data_location) + \
                   " --random_seed=" + str(self.args.random_seed) + \
                   " --params=" + str(self.args.params) + \
                   " --train_steps=" + str(self.args.train_steps) + \
                   " --steps_between_eval=" + str(self.args.steps_between_eval) + \
                   " --do_eval=" + str(self.args.do_eval) + \
                   " --save_checkpoints=" + str(self.args.save_checkpoints) + \
                   " --save_profile=" + str(self.args.save_profile) + \
                   " --print_iter=" + str(self.args.print_iter) + \
                   " --inter_op_parallelism_threads=" + \
                   str(self.args.num_inter_threads) + \
                   " --intra_op_parallelism_threads=" + \
                   str(self.args.num_intra_threads) +  \
                   " --learning_rate=" + \
                   str(self.args.learning_rate) +    \
                   " --static_batch=" + \
                   str(self.args.static_batch)


        #Running on single socket
        self.cmd = self.cmd + run_script + cmd_args

    def run(self):
        original_dir = os.getcwd()
        #os.chdir(self.args.model_source_dir)
        self.run_command(self.cmd)

        os.chdir(original_dir)
