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

#
'''This script optimizes feature columns in the model by removing error handling
and redundant nodes. Flag wide_and_deep_large_ds should be enabled for the additional
optimization for wide_and_deep_large_ds_model which involves fusion of categorical 
and numeric columns'''

from __future__ import division
import os
import sys
import tensorflow as tf
import argparse
import numpy as np
from google.protobuf import text_format
from tensorflow.python.framework import graph_util, ops, graph_io
from tensorflow.core.framework import graph_pb2
from tensorflow.python.client import session

parser = argparse.ArgumentParser()
parser.add_argument('--input-graph', type=str,
                    help='full path of graph to be optimized',
                    dest='input_graph', required=True)
parser.add_argument('--output-graph', type=str,
                    help='name of optimized graph',
                    dest='output_graph', required=True)
parser.add_argument('--output-nodes', type=str,
                    help='Comma seperated list of ouput nodes: head/predictions/logistic,   \
                    init_all_tables', dest='output_nodes', required=True)
parser.add_argument('--wide_and_deep_large_ds', type=bool,
                    help='Enable this flag when optimizing wide_and_deep_large_ds model,    \
                    to fuse categorical,numeric columns',
                    dest='enable_column_fusion', default=False)
args = parser.parse_args()
output_nodes = args.output_nodes.split(",")
output_nodes = ["import/"+str(i) for i in output_nodes]
graph = ops.Graph()
graph_def = graph_pb2.GraphDef()
old_graph_def = graph_pb2.GraphDef()
old_graph_def1 = graph_pb2.GraphDef()
file_ext = os.path.splitext(args.input_graph)[1]
with open(args.input_graph, "rb") as f:
    if file_ext == ".pbtxt":
        text_format.Merge(f.read(), graph_def)
    else:
        graph_def.ParseFromString(f.read())
categorical_column_weights_list, embedding_column_weights_list = [], []
with graph.as_default():
    tf.import_graph_def(graph_def)
    old_graph_def = graph.as_graph_def()
    name_node_dict = dict()
    #This method optimizes tf.embedding_column and tf.categorical_column_with_hash_bucket
    def optimize_categorical_embedding_with_hash_bucket(nodename, gatherfound):
        if ':' in nodename:
            nodename = nodename.split(':')[0]
        node = name_node_dict[nodename]
        if gatherfound[0] == 1 and node.op == "StringToHashBucketFast":
            return node.name
        if node.op == "GatherV2" and "Unique" in node.input[1]:
            gatherfound[0] = 1
            res = optimize_categorical_embedding_with_hash_bucket(node.input[1], gatherfound)
            if res:
                node.input[1] = res
                if "embedding" in node.input[0]:
                    embedding_column_weights_list.append(node.input[0]+":0")
                else:
                    categorical_column_weights_list.append(node.input[0]+":0")
                return node.name
        for inputname in node.input:
            res = optimize_categorical_embedding_with_hash_bucket(inputname, gatherfound)
            if res:
                return res
        return None

    #This method optimizes tf.feature_column.bucketized_column
    def optimize_bucketized_column(nodename, gatherfound):
        if ':' in nodename:
            nodename = nodename.split(':')[0]
        node = name_node_dict[nodename]
        if gatherfound[0] == 1 and node.op == "Reshape" and "Bucketize" in node.input[0]:
            return node.name
        if node.op == "GatherV2" and "Unique" in node.input[1]:
            gatherfound[0] = 1
            res = optimize_bucketized_column(node.input[1], gatherfound)
            if res:
                node.input[1] = res
                return node.name
        for inputname in node.input:
            res = optimize_bucketized_column(inputname, gatherfound)
            if res:
                return res
        return None

    #This method optimizes tf.feature_column.crossed_column
    def optimize_crossed_column(nodename, gatherfound):
        if ':' in nodename:
            nodename = nodename.split(':')[0]
        node = name_node_dict[nodename]
        if gatherfound[0] == 1 and node.op == "Identity" and "SparseCross" in node.input[0]:
            return node.name
        elif gatherfound[0] == 1 and node.op == "Identity" and \
                ("hash_table_Lookup" in node.input[0] or "lookup" in node.input[0]):
            return node.name
        elif gatherfound[0] == 2 and node.op == "SparseFillEmptyRows" and \
                "GatherV2" in node.input[0]:
            return node.name
        elif gatherfound[0] == 2 and node.op == "GatherV2" and "Unique" in node.input[1] and \
                 "Identity" not in node.input[0]:
            res = optimize_crossed_column(node.input[1], gatherfound)
            if res:
                node.input[1] = res+":1"
                return node.name
        if  gatherfound[0] != 2 and node.op == "GatherV2" and "Unique" in node.input[1]:
            gatherfound[0] = 1
            res = optimize_crossed_column(node.input[1], gatherfound)
            if res:
                node.input[1] = res
                return node.name
        elif  gatherfound[0] == 2 and node.op == "Mul" and "GatherV2" in node.input[0]:
            res = optimize_crossed_column(node.input[0], gatherfound)
            if res:
                node.input[0] = res
                return node.name
        elif node.op == "SegmentSum" and "mul" in node.input[0]:
            gatherfound[0] = 2
            res = optimize_crossed_column(node.input[0], gatherfound)
            return node.name
        for inputname in node.input:
            res = optimize_crossed_column(inputname, gatherfound)
            if res:
                return res
        return None

    # This method optimizes tf.feature_column.categorical_column_with_identity
    def optimize_categorical_column_with_identity(nodename, gatherfound):
        if ':' in nodename:
            nodename = nodename.split(':')[0]
        node = name_node_dict[nodename]
        if gatherfound[0] == 1 and node.op == "LookupTableFindV2":
            return node.name
        if node.op == "GatherV2" and "Unique" in node.input[1]:
            gatherfound[0] = 1
            res = optimize_categorical_column_with_identity(node.input[1], gatherfound)
            if res:
                node.input[1] = res
                return node.name
        for inputname in node.input:
            res = optimize_categorical_column_with_identity(inputname, gatherfound)
            if res:
                return res
        return None

    # This method optimizes tf.feature_column.categorical_column_with_vocabulary_list
    def optimize_categorical_with_voc_list(nodename, gatherfound):
        if ':' in nodename:
            nodename = nodename.split(':')[0]
        node = name_node_dict[nodename]
        if gatherfound[0] == 1 and node.op == "Select" and "Add" in node.input[2] and \
                "hash_table_Lookup" in node.input[1]:
            return node.name
        if node.op == "GatherV2" and "Unique" in node.input[1]:
            gatherfound[0] = 1
            res = optimize_categorical_with_voc_list(node.input[1], gatherfound)
            if res:
                node.input[1] = res
                return node.name
        for inputname in node.input:
            res = optimize_categorical_with_voc_list(inputname, gatherfound)
            if res:
                return res
        return None

    #This method optimizes tf.feature_column.numeric_column
    def optimize_numeric(nodename):
        if ':' in nodename:
            nodename = nodename.split(':')[0]
        node = name_node_dict[nodename]
        if node.op == "Reshape" and ("ParseExample" in node.input[0] or "div" in node.input[0]):
            return node.input[0]
        elif node.op == "Reshape" and "Maximum" in node.input[0]:
            return node.input[0]

    '''This method does model specific optimization(wide_deep_large_ds). It fuses 26 categorical,
        embedding weights to one constant and expects fused normalized inputs to the 
        numeric and hashed inputs to categorical placeholders. It also replaces gatherv2 
        with gathernd to gather weights from fused weights constant'''
    def fuse_categorical_numeric_columns():
        new_categorical_placeholder = tf.compat.v1.placeholder(tf.int64, shape=(None, None),
                                                     name='new_categorical_placeholder')
        new_numeric_placeholder = tf.compat.v1.placeholder(tf.float32,
                                                 shape=(None, None),
                                                 name='new_numeric_placeholder')
        categorical_column_weights_list.sort()
        embedding_column_weights_list.sort()
        sess = session.Session()
        categorical_weights_constant, embedding_weights_constant = [], []
        list_of_indices = [i for i in range(1, 11)]+[0] + \
                          [i for i in range(12, 19)]+[11] + \
                          [i for i in range(19, 26)]
        with sess.as_default():
            for i in list_of_indices:
                weight = graph.get_tensor_by_name(categorical_column_weights_list[i])
                categorical_weights_constant.append(weight.eval())
            for i in embedding_column_weights_list:
                weight = graph.get_tensor_by_name(i)
                embedding_weights_constant.append(weight.eval())
        fused_categorical_weights_const = np.stack(categorical_weights_constant)
        full_weights_constant_categorical = tf.constant(fused_categorical_weights_const,
                                                        name='full_weights_constant_categorical')
        batch_gather_op = tf.gather_nd(full_weights_constant_categorical,
                                       new_categorical_placeholder,
                                       name='gather_categorical_weights')
        reshape_result = tf.reshape(batch_gather_op, shape=[-1, 26])
        reduce_sum_op = tf.reduce_sum(reshape_result, 1, keepdims=True)
        fused_embedding_weights_const = np.stack(embedding_weights_constant)
        full_weights_constant_embedding = tf.constant(fused_embedding_weights_const,
                                                      name='full_weights_constant_embedding')
        batch_gather_op_embedding = tf.gather_nd(full_weights_constant_embedding,
                                                 new_categorical_placeholder,
                                                 name='gather_embedding_weights')
        embedding_reshape = tf.reshape(batch_gather_op_embedding,
                                       shape=[-1, 32*26],
                                       name='embedding_reshape')
        real_div_input_tens_list = [embedding_reshape, new_numeric_placeholder]
        new_concat_node = tf.concat(real_div_input_tens_list, name='new_concat_node', axis=1)
        concat_tensor = graph.get_tensor_by_name("new_concat_node:0")


    '''Parsing all the nodes of graph and identifying feature columns to optimize '''
    for node in old_graph_def.node:
        nodename = node.name
        if node.op == "ConcatV2" and "dnn/input_from_feature_columns" in nodename and \
                       "input_layer/concat" in nodename:
            dnn_concat_node = node
        elif node.op == "AddN" and "weighted_sum_no_bias" in nodename:
            weightsumnobias_node = node
        name_node_dict[nodename] = node
    gatherfound = [0]
    try:
        for i, inputname in enumerate(weightsumnobias_node.input):
            if  'weighted_by' not in inputname and '_X_' not in inputname:
                gatherfound[0] = 0
                res = optimize_categorical_with_voc_list(weightsumnobias_node.input[i], gatherfound)
                if res:
                    weightsumnobias_node.input[i] = res
                else:
                    gatherfound[0] = 0
                    res = optimize_categorical_column_with_identity(weightsumnobias_node.input[i],
                                                                    gatherfound)
                    if res:
                        weightsumnobias_node.input[i] = res
                    else:
                        gatherfound[0] = 0
                        res = optimize_categorical_embedding_with_hash_bucket(
                            weightsumnobias_node.input[i], gatherfound)
                        if res:
                            weightsumnobias_node.input[i] = res
                        else:
                            gatherfound[0] = 0
                            res = optimize_bucketized_column(weightsumnobias_node.input[i], gatherfound)
                            if res:
                                weightsumnobias_node.input[i] = res
            elif '_X_' in inputname or 'weighted_by' in inputname:
                gatherfound[0] = 0
                res = optimize_crossed_column(weightsumnobias_node.input[i], gatherfound)
                if res:
                    weightsumnobias_node.input[i] = res

        for i, inputname in enumerate(dnn_concat_node.input):
            if '_embedding' in inputname and 'shared_embedding' not in inputname \
                and 'weighted_by' not in inputname and '_X_' not in inputname:
                gatherfound[0] = 0
                res = optimize_categorical_with_voc_list(dnn_concat_node.input[i], gatherfound)
                if res:
                    dnn_concat_node.input[i] = res
                else:
                    gatherfound[0] = 0
                    res = optimize_categorical_column_with_identity(
                        dnn_concat_node.input[i], gatherfound)
                    if res:
                        dnn_concat_node.input[i] = res
                    else:
                        gatherfound[0] = 0
                        res = optimize_categorical_embedding_with_hash_bucket(
                            dnn_concat_node.input[i], gatherfound)
                        if res:
                            dnn_concat_node.input[i] = res

            elif 'shared_embedding' not in inputname:
                res2 = optimize_numeric(dnn_concat_node.input[i])
                if res2:
                    dnn_concat_node.input[i] = res2
            else:
                gatherfound[0] = 0
                #shared_embedding
                res = optimize_crossed_column(dnn_concat_node.input[i], gatherfound)
                if res:
                    dnn_concat_node.input[i] = res
        if args.enable_column_fusion:
            fuse_categorical_numeric_columns()
            old_graph_def = graph.as_graph_def()
            for node in old_graph_def.node:
                if node.name == "new_concat_node":
                    node.input[1] = "new_numeric_placeholder:0"
                elif node.op == "BiasAdd" and "linear_model/weighted_sum" in node.name:
                    node.input[0] = "Sum:0"
                elif  node.op == "MatMul" and "hiddenlayer_0/MatMul" in node.name:
                    node.input[0] = "new_concat_node:0"
    except Exception as e:
        print(e)
        print('--------------------------------------------------------------------------')
        print("Cannot optimize the given graph. The given graph might be an optimized one")
        print('--------------------------------------------------------------------------')
        sys.exit()             

new_graph_def = tf.compat.v1.GraphDef()
new_graph_def = tf.compat.v1.graph_util.extract_sub_graph(
    old_graph_def,
    output_nodes
)

filename = args.output_graph
graph_io.write_graph(new_graph_def,
                     os.path.dirname(filename),
                     os.path.basename(filename),
                     as_text=False)
print('Optimized graph created')
