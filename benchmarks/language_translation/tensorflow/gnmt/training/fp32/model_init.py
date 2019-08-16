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
    """Model initializer for GNMT FP32 training"""

    def __init__(self, args, custom_args=[], platform_util=None):
        super(ModelInitializer, self).__init__(args, custom_args, platform_util)

        # Set KMP env vars, if they haven't already been set
        config_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        self.set_kmp_vars(config_file_path)

        # set num_inter_threads and num_intra_threads (override inter threads to 2)
        self.set_num_inter_intra_threads()

        DATA_DIR = os.path.join(self.args.intelai_models, self.args.mode,
                                self.args.precision, "wmt16")
        HPARAM_DIR = os.path.join(self.args.intelai_models, self.args.mode,
                                  self.args.precision, "standard_hparams")

        set_env_var("OMP_NUM_THREADS", self.args.num_intra_threads)

        if self.args.data_location is None:
            data_dir = DATA_DIR
        else:
            data_dir = self.args.data_location
        # Train parater control
        arg_parser = ArgumentParser(description="process custom_args")
        arg_parser.add_argument(
            "-s", "--src", help="source lanuage",
            dest="src", default="de")
        arg_parser.add_argument(
            "-t", "--tgt", help="target lanuage",
            dest="tgt", default="en")
        arg_parser.add_argument(
            "-VP", "--vocab_prefix",
            help="prefix of vocabulary file",
            dest="vocab_prefix",
            default=data_dir + "/vocab.bpe.32000")
        arg_parser.add_argument(
            "-TP", "--train_prefix",
            help="prefix of train file",
            dest="train_prefix",
            default=data_dir + "/train.tok.clean.bpe.32000")
        arg_parser.add_argument(
            "-DP", "--dev_prefix",
            help="prefix of dev file",
            dest="dev_prefix",
            default=data_dir + "/newstest2013.tok.bpe.32000")
        arg_parser.add_argument(
            "-TSP", "--test_prefix",
            help="prefix of test file",
            dest="test_prefix",
            default=data_dir + "/newstest2015.tok.bpe.32000")
        arg_parser.add_argument(
            "-OD", "--output_dir",
            help="output directory",
            dest="output_dir",
            default=self.args.output_dir)
        arg_parser.add_argument(
            "-NU", "--num_units",
            help="number of units",
            dest="num_units",
            default=1024)
        arg_parser.add_argument(
            "-DO", "--dropout",
            help="dropout",
            dest="dropout",
            default=0.2)
        arg_parser.add_argument(
            "-BS", "--batch_size",
            help="batch size",
            dest="batch_size",
            default=512)
        arg_parser.add_argument(
            "-NP", "--num_processes",
            help="number of processes",
            dest="num_processes",
            default=2)
        arg_parser.add_argument(
            "-NPPN", "--num_processes_per_node",
            help="number of processes per node",
            dest="num_processes_per_node",
            default=1)
        arg_parser.add_argument(
            "-NT", "--num_inter_threads",
            help="number of inter threads",
            dest="num_inter_threads",
            default=1)
        arg_parser.add_argument(
            "-NAT", "--num_intra_threads",
            help="number of intra threads",
            dest="num_intra_threads",
            default=28)
        arg_parser.add_argument(
            "-NTS", "--num_train_steps",
            help="number of train steps",
            dest="num_train_steps",
            default=340000)
        arg_parser.add_argument(
            "-HPMS", "--hparams_path", help="hparameter files location",
            dest="hparams_path",
            default=HPARAM_DIR + "/wmt16_gnmt_4_layer_multi_instances.json")
        self.args = arg_parser.parse_args(self.custom_args, namespace=self.args)

        # Model parameter control

        cmd_args = " --src=" + self.args.src + " --tgt=" + self.args.tgt + \
                   " --vocab_prefix=" + os.path.join(data_dir, self.args.vocab_prefix) + \
                   " --train_prefix=" + os.path.join(data_dir, self.args.train_prefix) + \
                   " --dev_prefix=" + os.path.join(data_dir, self.args.dev_prefix) + \
                   " --test_prefix=" + os.path.join(data_dir, self.args.test_prefix) + \
                   " --out_dir=" + self.args.output_dir + \
                   " --num_units=" + str(self.args.num_units) + \
                   " --dropout=" + str(self.args.dropout) + \
                   " --batch_size=" + str(self.args.batch_size) + \
                   " --num_inter_threads=" + \
                   str(self.args.num_inter_threads) + \
                   " --num_intra_threads=" + \
                   str(self.args.num_intra_threads) + \
                   " --num_train_steps=" + str(self.args.num_train_steps) + \
                   " --hparams_path=" + self.args.hparams_path

        self.run_script_dir = os.path.join(self.args.intelai_models, self.args.mode, self.args.precision, "nmt")
        multi_instance_param_list = ["-genv:I_MPI_ASYNC_PROGRESS=1",
                                     "-genv:I_MPI_FABRICS=shm",
                                     "-genv:I_MPI_PIN_DOMAIN=socket",
                                     "-genv:I_MPI_ASYNC_PROGRESS_PIN={},{}".format(0, self.args.num_intra_threads),
                                     "-genv:OMP_NUM_THREADS={}".format(self.args.num_intra_threads)]
        self.cmd = self.get_multi_instance_train_prefix(multi_instance_param_list)
        self.cmd += "{} ".format(self.python_exe)
        run_script = "-m  nmt.nmt "
        self.cmd = self.cmd + run_script + cmd_args

    def run(self):
        if self.cmd:
            # The generate.py script expects that we run from the model source
            # directory.  Save off the current working directory so that we can
            # restore it when the script is done.
            original_dir = os.getcwd()
            os.chdir(self.run_script_dir)
            # Run benchmarking
            self.run_command(self.cmd)
            # Change context back to the original dir
            os.chdir(original_dir)
