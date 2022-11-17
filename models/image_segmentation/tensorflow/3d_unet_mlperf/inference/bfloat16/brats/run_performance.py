#
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

#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from argparse import ArgumentParser
import os
import pickle
import sys
import math

import numpy as np
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
from tensorflow.python.framework import dtypes
from tensorflow.core.protobuf import rewriter_config_pb2

INPUTS = 'input'
OUTPUTS = 'Identity'

class unet_3d_tf:
    """Evaluate 3d_unet with optimized TensorFlow graph"""

    def __init__(self):
        arg_parser = ArgumentParser(description='Parse args')

        arg_parser.add_argument('-e', "--num-inter-threads",
                                help='The number of inter-thread.',
                                dest='num_inter_threads', type=int, default=0)
        arg_parser.add_argument('-a', "--num-intra-threads",
                                help='The number of intra-thread.',
                                dest='num_intra_threads', type=int, default=0)
        arg_parser.add_argument('-m', "--model-name",
                                help='Specify the model name to run benchmark for',
                                dest='model_name')
        arg_parser.add_argument('-g', "--input-graph",
                                help='Specify the input graph for the transform tool',
                                dest='input_graph')
        arg_parser.add_argument("--results-file-path",
                                help="File path for the inference results",
                                dest="results_file_path", default=None)
        arg_parser.add_argument("--warmup-steps", type=int, default=10,
                                help="number of warmup steps")
        arg_parser.add_argument("--steps", type=int, default=50,
                                help="number of steps")
        arg_parser.add_argument("--batch-size", type=int, default=1)

        self.args = arg_parser.parse_args()
        print (self.args)

    def run(self):
        print("Run inference")
        graph = tf.Graph()
        with graph.as_default():
            graph_def = tf.compat.v1.GraphDef()
            with open(self.args.input_graph, "rb") as f:
                graph_def.ParseFromString(f.read())
            output_graph = optimize_for_inference(graph_def, [INPUTS], [OUTPUTS],
                                dtypes.float32.as_datatype_enum, False)
            tf.import_graph_def(output_graph, name="")

        input_tensor = graph.get_tensor_by_name('input:0')
        output_tensor = graph.get_tensor_by_name('Identity:0')
        batch_size = self.args.batch_size
        warmup_steps = self.args.warmup_steps
        steps = self.args.steps

        if input_tensor.op.type != 'Placeholder':
            raise ValueError('input_layer should of type Placeholder')
        else:
            input_shape = [batch_size, 4, 224, 224, 160]
        print("Input shape {}".format(input_shape))

        input_data = np.random.randn(*input_shape).astype(np.float32)
        config = tf.compat.v1.ConfigProto()
        config.intra_op_parallelism_threads=self.args.num_intra_threads
        config.inter_op_parallelism_threads=self.args.num_inter_threads
        config.graph_options.rewrite_options.auto_mixed_precision_mkl = rewriter_config_pb2.RewriterConfig.ON
        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            print("Started warmup for {} steps...".format(warmup_steps))
            for j in range(warmup_steps):
                _ = sess.run(output_tensor, {input_tensor: input_data})
            print("Warmup done.")
            print("Started benchmark for {} steps...".format(steps))
            print("Inference with dummy data")
            start = time.time()
            for i in range(steps):
                if i%10 == 0:
                    print("Iteration {}".format(i))
                output = sess.run(output_tensor, {input_tensor: input_data})
            end = time.time()
            average_time = (end - start)/steps
            throughput = batch_size/average_time
            print("Average time for step: {} sec".format(average_time) )
            print("Throughput: {} samples/sec".format(throughput))
            if (self.args.batch_size == 1):
                print('Latency: %.3f ms' % (average_time * 1000))

if __name__ == "__main__":
    evaluate_opt_graph = unet_3d_tf()
    evaluate_opt_graph.run()

