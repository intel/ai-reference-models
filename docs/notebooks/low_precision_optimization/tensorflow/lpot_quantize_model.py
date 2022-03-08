#!/usr/bin/env python
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

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import logging

tf.get_logger().setLevel(logging.ERROR)
tf.compat.v1.disable_eager_execution()

import lpot
import sys 


def save_int8_frozen_pb(q_model, path):
    from tensorflow.python.platform import gfile
    f = gfile.GFile(path, 'wb')
    f.write(q_model.as_graph_def().SerializeToString())
    print("Save to {}".format(path))

def auto_tune(input_graph_path, yaml_config):
    quan = lpot.Quantization(yaml_config)
    q_model = quan(
        input_graph_path,
        q_dataloader = None,
        eval_func=None,
        eval_dataloader = None
        )
    return q_model

def check_dataset_folder(folder):
    if os.path.exists(folder) and not os.path.isfile(folder): 
        files = os.listdir(folder) 
        return len(files)>0

    return False

def quantize():
    yaml_file = "resnet50_v1.yaml"
    fp32_frozen_pb_file = "resnet50_fp32_pretrained_model.pb"
    int8_pb_file = "resnet50_int8_model.pb"

    dataset_folder = "tf_2012_val"
    if not check_dataset_folder(dataset_folder):
        print("Error, folder {} is empty directory or not existed".format(dataset_folder))
        return 1

    q_model = auto_tune(fp32_frozen_pb_file, yaml_file)
    save_int8_frozen_pb(q_model, int8_pb_file)

    return 0

if __name__=="__main__":

    res = quantize()
    sys.exit(res)
