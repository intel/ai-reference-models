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
# SPDX-License-Identifier: EPL-2.0
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
tf.compat.v1.disable_v2_behavior()
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
parser.add_argument('--num_parallel_batches', type=int,
                    help='number of parallel batches',
                    required=False,
                    default=1,
                    dest='num_parallel_batches')
parser.add_argument("--accuracy_only", type=str2bool,
                    nargs='?', const=True, default=False,
                    dest='compute_accuracy', required=False,
                    help="Enable accuracy calculation")

args = parser.parse_args()
if args.num_omp_threads:
    os.environ["OMP_NUM_THREADS"] = args.num_omp_threads
num_parallel_batches = args.num_parallel_batches
output_probabilities_node = 'import/import/head/predictions/probabilities'
while_probabilities_node = 'while/import/'+output_probabilities_node+':0'
while_softmax_operation = 'while/import/'+output_probabilities_node
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

numeric_feature_names = ["numeric_1"]
string_feature_names = ["string_1"]
if args.compute_accuracy:
    full_features_names = numeric_feature_names + string_feature_names + ["label"]
    feature_datatypes = [tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0.0, allow_missing=True)]+[tf.io.FixedLenSequenceFeature(
            [], tf.int64, default_value=0, allow_missing=True)]+[tf.io.FixedLenSequenceFeature([], tf.int64, default_value=0, allow_missing=True)]
else:
    full_features_names = numeric_feature_names + string_feature_names
    feature_datatypes = [tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0.0, allow_missing=True)]+[tf.io.FixedLenSequenceFeature(
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
    dataset = dataset.cache()
    dataset = dataset.prefetch(1)
    dataset = dataset.repeat()
    return dataset


data_file = args.data_location
no_of_test_samples = sum(1 for _ in tf.compat.v1.python_io.tf_record_iterator(data_file))
no_of_batches = math.ceil(float(no_of_test_samples)/batch_size)
with graph.as_default():
    tf.import_graph_def(graph_def)
    res_dataset = input_fn(data_file, 1, False, batch_size)
    iterator = tf.compat.v1.data.make_one_shot_iterator(res_dataset)
    next_element = iterator.get_next()
    iterator_names = [i.name.split(':')[1] for i in next_element]
    placeholder_expandims = {}
    full_nodes = []
    old_graph_def = graph.as_graph_def()
    for node in old_graph_def.node:
        k = node.name
        if k == "IteratorGetNext":
            iterator_node = node
        elif (node.op == "GatherNd" or node.op == 'ConcatV2') and (placeholder_name in node.input[1] or categorical_placeholder in node.input[1]):
            if node.op == 'GatherNd' and node.name == 'import/gather_categorical_weights':
                gather_categorical_node = node
            elif node.op == 'GatherNd' and node.name == 'import/gather_embedding_weights':
                gather_embedding_node = node
            elif node.op == 'ConcatV2':
                concat_node = node

    gather_categorical_node.input[1] = iterator_node.name+":1"
    gather_embedding_node.input[1] = iterator_node.name+":1"
    concat_node.input[1] = iterator_node.name+":0"


new_graph_def = tf.compat.v1.GraphDef()
new_graph_def = tf.compat.v1.graph_util.extract_sub_graph(
    old_graph_def,
    [output_probabilities_node]
)
tf.compat.v1.reset_default_graph()
graph = ops.Graph()

with graph.as_default():
    i = tf.constant(0)
    arr = tf.TensorArray(dtype=tf.int32, size=2000, dynamic_size=True)
    def _body(i, arr):
        tf.import_graph_def(new_graph_def)
        output_tensor = graph.get_tensor_by_name(while_probabilities_node)
        if args.compute_accuracy:
            labels_tensor = graph.get_tensor_by_name("while/import/IteratorGetNext:2")
            predicted_labels = tf.argmax(output_tensor,1,output_type=tf.int64)
            correctly_predicted_bool = tf.equal(predicted_labels, labels_tensor)
            num_correct_predictions_batch = tf.reduce_sum(tf.cast(correctly_predicted_bool, tf.int32))
        else:
            predicted_labels = tf.argmax(output_tensor,1,output_type=tf.int32)
            num_correct_predictions_batch = tf.reduce_sum(predicted_labels)
        arr = arr.write(i, num_correct_predictions_batch)
        i = tf.add(i, 1)
        return i, arr
    i, arr = tf.compat.v2.while_loop(cond=lambda i, x: i < int(no_of_batches), body=_body, loop_vars=[i, arr], parallel_iterations=num_parallel_batches)
    array_gather = arr.gather(tf.range(0, int(no_of_batches), delta=1, dtype=None, name='range'))

with tf.compat.v1.Session(config=config, graph=graph) as sess:
    num_correct_predictions_batch = sess.run(array_gather)
    inference_start = time.time()
    try:
        num_correct_predictions_batch = sess.run(array_gather)
    except Exception as e:
        print('--------------------------------------------------')
        print("Exception during execution of model:  ",e)
        print('--------------------------------------------------')
        sys.exit()
    total_num_correct_predictions = num_correct_predictions_batch.sum(axis=0)
    inference_end = time.time()
if args.compute_accuracy:
    accuracy = (float(total_num_correct_predictions)/float(no_of_test_samples))
evaluate_duration = inference_end - inference_start
latency = (1000 * float(batch_size * num_parallel_batches) * float(evaluate_duration) / float(no_of_test_samples))

throughput = no_of_test_samples/evaluate_duration
print('--------------------------------------------------')
print('Total test records           : ', no_of_test_samples)
print('Batch size is                : ', batch_size)
print('Number of batches            : ', int(no_of_batches))
if args.compute_accuracy:
    print('Classification accuracy (%)  : ', round((accuracy * 100), 4))
    print('No of correct predictions    : ', int(total_num_correct_predictions))
print('Inference duration (seconds) : ', round(evaluate_duration, 4))
print('Average Latency (ms/batch)   : ', round(latency,4))
print('Throughput is (records/sec)  : ', round(throughput, 3))
print('--------------------------------------------------')
