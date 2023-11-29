#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Intel Corporation
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
import sys
import os
import numpy as np
import argparse
import collections
import time
import math
import json
import datetime

import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.core.framework import graph_pb2
from google.protobuf import text_format


def str2bool(v):
    if v.lower() in ('true'):
        return True
    else:
        return False


parser = argparse.ArgumentParser()
parser.add_argument('--input_graph', type=str,
                    help='file name for graph',
                    dest='input_graph',
                    required=True)
parser.add_argument('--data_location', type=str,
                    help='full path of data file',
                    dest='data_location',
                    required=True)
parser.add_argument('--batch_size', type=int,
                    help='batch size for inference.Default is 512',
                    default=512,
                    dest='batch_size')
parser.add_argument('--iterations', type=int,
                    help='number of iterations to run in benchmark mode',
                    default=2000,
                    dest='iterations')
parser.add_argument('--num_intra_threads', type=int,
                    help='number of threads for an operator',
                    required=False,
                    default=28,
                    dest='num_intra_threads')
parser.add_argument('--num_inter_threads', type=int,
                    help='number of threads across operators',
                    required=False,
                    default=2,
                    dest='num_inter_threads')
parser.add_argument('--num_omp_threads', type=str,
                    help='number of threads to use',
                    required=False,
                    default=None,
                    dest='num_omp_threads')
parser.add_argument("--accuracy_only", type=str2bool,
                    nargs='?', const=True, default=False,
                    dest='compute_accuracy', required=False,
                    help="Enable accuracy calculation")

args = parser.parse_args()
if args.num_omp_threads:
    os.environ["OMP_NUM_THREADS"] = args.num_omp_threads

output_probabilities_node = 'import/import/head/predictions/probabilities'
probabilities_node = 'import/' + output_probabilities_node + ':0'
placeholder_name = 'import/new_numeric_placeholder'
categorical_placeholder = 'import/new_categorical_placeholder'

config = tf.compat.v1.ConfigProto(log_device_placement=False,
                                  inter_op_parallelism_threads=args.num_inter_threads,
                                  intra_op_parallelism_threads=args.num_intra_threads)
graph = ops.Graph()
graph_def = graph_pb2.GraphDef()

filename, file_ext = os.path.splitext(args.input_graph)

batch_size = args.batch_size
with open(args.input_graph, "rb") as f:
    if file_ext == ".pbtxt":
        text_format.Merge(f.read(), graph_def)
    else:
        graph_def.ParseFromString(f.read())
with graph.as_default():
    tf.import_graph_def(graph_def)
numeric_feature_names = ["numeric_1"]
string_feature_names = ["string_1"]
if args.compute_accuracy:
    full_features_names = numeric_feature_names + string_feature_names + ["label"]
    feature_datatypes = [tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0.0, allow_missing=True)] + [
        tf.io.FixedLenSequenceFeature(
            [], tf.int64, default_value=0, allow_missing=True)] + [
                            tf.io.FixedLenSequenceFeature([], tf.int64, default_value=0, allow_missing=True)]
else:
    full_features_names = numeric_feature_names + string_feature_names
    feature_datatypes = [tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0.0, allow_missing=True)] + [
        tf.io.FixedLenSequenceFeature(
            [], tf.int64, default_value=0, allow_missing=True)]


def input_fn(data_file, num_epochs, shuffle, batch_size):
    """Generate an input function for the Estimator."""

    def _parse_function(proto):
        f = collections.OrderedDict(
            zip(full_features_names, feature_datatypes))
        parsed_features = tf.io.parse_example(proto, f)
        parsed_feature_vals_num = [tf.reshape(
            parsed_features["numeric_1"], shape=[-1, 13])]
        parsed_feature_vals_str = [tf.reshape(
            parsed_features["string_1"], shape=[-1, 2]) for i in string_feature_names]
        parsed_feature_vals = parsed_feature_vals_num + parsed_feature_vals_str
        if args.compute_accuracy:
            parsed_feature_vals_label = [tf.reshape(parsed_features[i], shape=[-1]) for i in ["label"]]
            parsed_feature_vals = parsed_feature_vals + parsed_feature_vals_label
        return parsed_feature_vals

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TFRecordDataset([data_file])
    if shuffle:
        dataset = dataset.shuffle(buffer_size=20000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(_parse_function, num_parallel_calls=28)
    dataset = dataset.prefetch(batch_size * 10)
    return dataset


data_file = args.data_location
no_of_test_samples = sum(1 for _ in tf.compat.v1.python_io.tf_record_iterator(data_file))
no_of_batches = math.ceil(float(no_of_test_samples) / batch_size)
placeholder_list = ['import/new_numeric_placeholder:0', 'import/new_categorical_placeholder:0']
input_tensor = [graph.get_tensor_by_name(name) for name in placeholder_list]
output_name = "import/head/predictions/probabilities"
output_tensor = graph.get_tensor_by_name("import/" + output_name + ":0")
correctly_predicted = 0
total_infer_consume = 0.0
warm_iter = 100
features_list = []
with tf.compat.v1.Session(config=config, graph=graph) as sess:
    res_dataset = input_fn(data_file, 1, False, batch_size)
    iterator = tf.compat.v1.data.make_one_shot_iterator(res_dataset)
    next_element = iterator.get_next()
    for i in range(int(no_of_batches)):
        batch = sess.run(next_element)
        features = batch[0:3]
        features_list.append(features)

with tf.compat.v1.Session(config=config, graph=graph) as sess1:
    i = 0
    if not args.compute_accuracy:
        no_of_batches = args.iterations
        no_of_test_samples = no_of_batches * batch_size
    while True:
        if i >= no_of_batches:
            break
        if i > warm_iter:
            inference_start = time.time()
        logistic = None
        if args.compute_accuracy:
            logistic = sess1.run(output_tensor, dict(zip(input_tensor, features_list[i][0:2])))
        else:
            logistic = sess1.run(output_tensor, dict(zip(input_tensor, features_list[0][0:2])))
        if i > warm_iter:
            infer_time = time.time() - inference_start
            total_infer_consume += infer_time
        if args.compute_accuracy:
            predicted_labels = np.argmax(logistic, 1)
            correctly_predicted = correctly_predicted + np.sum(features_list[i][2] == predicted_labels)

        i = i + 1
    inference_end = time.time()
if args.compute_accuracy:
    accuracy = (
            float(correctly_predicted) / float(no_of_test_samples))
evaluate_duration = total_infer_consume
latency = (1000 * batch_size * float(evaluate_duration) / float(no_of_test_samples - warm_iter * batch_size))
throughput = (no_of_test_samples - warm_iter * batch_size) / evaluate_duration

print('--------------------------------------------------')
print('Total test records           : ', no_of_test_samples)
print('Batch size is                : ', batch_size)
print('Number of batches            : ', int(no_of_batches))
if args.compute_accuracy:
    print('Classification accuracy (%)  : ', round((accuracy * 100), 4))
    print('No of correct predictions    : ', int(correctly_predicted))
print('Inference duration (seconds) : ', round(evaluate_duration, 4))
print('Average Latency (ms/batch)   : ', round(latency, 4))
print('Throughput is (records/sec)  : ', round(throughput, 3))
print('--------------------------------------------------')