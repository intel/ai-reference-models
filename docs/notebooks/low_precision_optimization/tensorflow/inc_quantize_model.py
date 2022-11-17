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

import neural_compressor as inc
print("neural_compressor version {}".format(inc.__version__))


import tensorflow as tf
print("tensorflow {}".format(tf.__version__))

import logging
tf.get_logger().setLevel(logging.ERROR)
tf.compat.v1.disable_eager_execution()

from neural_compressor.experimental import Quantization, common
import logging
import sys 

def save_frozen_pb(model, mod_path):
    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda x: model(x))
    concrete_function = full_model.get_concrete_function(
        x=tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_model = convert_variables_to_constants_v2(concrete_function)

    # Generate frozen pb
    tf.io.write_graph(graph_or_graph_def=frozen_model.graph,
                      logdir=".",
                      name=mod_path,
                      as_text=False)
    
def ver2int(ver):
    s_vers = ver.split(".")
    res = 0
    for i, s in enumerate(s_vers):
        res += int(s)*(100**(2-i))

    return res

def compare_ver(src, dst):
    src_ver = ver2int(src)
    dst_ver = ver2int(dst)
    if src_ver>dst_ver:
        return 1
    if src_ver<dst_ver:
        return -1
    return 0

def auto_tune(input_graph_path, yaml_config): 
    quantizer = Quantization(yaml_config)
    quantizer.model = common.Model(input_graph_path)
    q_model = quantizer.fit()
    if compare_ver(inc.__version__, "1.9")>=0:
        q_model = quantizer.fit()
    else:
        q_model = quantizer()        
        
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
    q_model.save(int8_pb_file)

    return 0

if __name__=="__main__":

    res = quantize()
    sys.exit(res)
