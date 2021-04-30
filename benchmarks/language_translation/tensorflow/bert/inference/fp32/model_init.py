#  #
#  -*- coding: utf-8 -*-
#  #
#  Copyright (c) 2019 Intel Corporation
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#     http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  #
#  SPDX-License-Identifier: EPL-2.0
#  #
#

import os
import argparse

from common.base_model_init import BaseModelInitializer, set_env_var


class ModelInitializer(BaseModelInitializer):
    """model initializer for BERT fp32 inference"""

    def __init__(self, args, custom_args, platform_util):
        super(ModelInitializer, self).__init__(args, custom_args, platform_util)

        self.args = args
        self.custom_args = custom_args
        self.platform_util = platform_util

        # set KMP env vars, if they haven't already been set
        config_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        self.set_kmp_vars(config_file_path)

        parser = argparse.ArgumentParser()
        parser.add_argument('--task_name', type=str, default='XNLI', dest='task_name', help='take name')
        parser.add_argument('--max_seq_length', type=int, default=128, dest='max_seq_length',
                            help='max sequence length')
        parser.add_argument('--learning_rate', type=float, default=5e-5, dest='learning_rate', help='learning rate')
        self.args = parser.parse_args(self.custom_args, namespace=self.args)

        self.set_num_inter_intra_threads(num_inter_threads=self.args.num_inter_threads,
                                         num_intra_threads=self.args.num_intra_threads)

        set_env_var("OMP_NUM_THREADS", self.args.num_intra_threads)

        # model parameter control
        script_path = os.path.join(self.args.intelai_models, self.args.mode, self.args.precision, "run_classifier.py")
        self.run_cmd = self.get_command_prefix(self.args.socket_id) + "{} {}".format(self.python_exe, script_path)
        self.run_cmd += " --data_dir={} --output_dir={}".format(self.args.data_location, format(self.args.output_dir))
        vocab_file = os.path.join(self.args.checkpoint, "vocab.txt")
        bert_config_file = os.path.join(self.args.checkpoint, "bert_config.json")
        init_checkpoint = os.path.join(self.args.checkpoint, "bert_model.ckpt")
        self.run_cmd += " --vocab_file={} --bert_config_file={} --init_checkpoint={}".format(
            vocab_file, bert_config_file, init_checkpoint)
        self.run_cmd += " --task_name={} --max_seq_length={} --eval_batch_size={} --learning_rate={}".format(
            self.args.task_name, self.args.max_seq_length, self.args.batch_size, self.args.learning_rate)
        self.run_cmd += " --num_inter_threads={} --num_intra_threads={}".format(
            self.args.num_inter_threads, self.args.num_intra_threads)
        self.run_cmd += " --do_eval=true"

        # Test accuracy if needed
        if self.args.accuracy_only:
            self.run_cmd += " --do_train=true --num_train_epochs=3.0 --train_batch_size=32"
        else:
            self.run_cmd += " --do_train=false"

    def run(self):
        if self.run_cmd:
            self.run_command(self.run_cmd)
