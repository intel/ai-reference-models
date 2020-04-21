#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2018-2020 Intel Corporation
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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from common.base_model_init import BaseModelInitializer

import os


import argparse



class ModelInitializer(BaseModelInitializer):
    """Model initializer for minigo"""

    def __init__(self, args, custom_args=[], platform_util=None):
        super(ModelInitializer, self).__init__(args, custom_args, platform_util)

        arg_parser = argparse.ArgumentParser(description='Parse additional args')
        
        arg_parser.add_argument(
            "--quantization", help="quantization flag",
            dest="quantization", default="False")     
        arg_parser.add_argument(
            "--large-scale", help="train on large scale",
            dest="large_scale", default="False") 
        arg_parser.add_argument(
            "--num-train-nodes", help="number of train nodes",
            dest="num_train_nodes", default=0, type=int) 
        arg_parser.add_argument(
            "--num-eval-nodes", help="number of evaluation nodes",
            dest="num_eval_nodes", default=0, type=int)  
        arg_parser.add_argument(
            "--multi-node", help="train on large scale",
            dest="multi_node", default="False")    
        
        self.additional_args, unknown_args = arg_parser.parse_known_args(custom_args)
        
        if self.additional_args.large_scale == "True" and self.additional_args.multi_node == "True":
            # multi-node training mode with large scale
            self.cmd = "./run_mn.sh "
            self.cmd += " {0}".format(self.additional_args.num_train_nodes)
            self.cmd += " {0}".format(self.additional_args.num_eval_nodes)
            self.cmd += " {0}".format(self.additional_args.quantization)
        elif self.additional_args.large_scale == "False" and self.additional_args.multi_node == "True":
            # multi-node training mode
            self.cmd = "./run_mn.sh "
            self.cmd += " {0}".format(self.additional_args.num_train_nodes)
            self.cmd += " {0}".format(self.additional_args.quantization)
        else:
            # single-node training mode
            self.cmd = "./run.sh "
            self.cmd += " {0}".format(self.additional_args.quantization)
            
    def run(self):
        org_path = os.getcwd()
        os.chdir(self.args.model_source_dir)
        self.run_command(self.cmd)
        os.chdir(org_path)
