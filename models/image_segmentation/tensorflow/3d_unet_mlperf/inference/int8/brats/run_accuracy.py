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
import array

import numpy as np
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
from tensorflow.python.framework import dtypes
from tensorflow.core.protobuf import rewriter_config_pb2

from nnunet.evaluation.region_based_evaluation import evaluate_regions, get_brats_regions

from inference.nnUNet.setup import setup
from inference.nnUNet.postprocess import postprocess_output

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
                                help='Specify the input graph.',
                                dest='input_graph')
        arg_parser.add_argument('-d', "--data-location",
                                help='Specify the location of the data.',
                                dest="data_location", default=None)
        arg_parser.add_argument('-r', "--accuracy-only",
                                help='For accuracy measurement only.',
                                dest='accuracy_only', action='store_true')
        arg_parser.add_argument("--batch-size", type=int, default=1)
        arg_parser.add_argument('--onednn-graph', dest='onednn_graph',
                                help='enable OneDNN Graph',
                                action='store_true')

        self.args = arg_parser.parse_args()
        print(self.args)

    def run(self):
        print("Run inference for accuracy")
        setup(self.args.data_location, self.args.input_graph)

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

        config = tf.compat.v1.ConfigProto()
        config.intra_op_parallelism_threads=self.args.num_intra_threads
        config.inter_op_parallelism_threads=self.args.num_inter_threads

        if self.args.onednn_graph:
            import intel_extension_for_tensorflow as itex
            auto_mixed_precision_options = itex.AutoMixedPrecisionOptions()
            auto_mixed_precision_options.data_type = itex.BFLOAT16

            auto_mixed_precision_options.allowlist_add= "Rsqrt,SquaredDifference,Mean"
            auto_mixed_precision_options.denylist_remove = "Mean"
            auto_mixed_precision_options.denylist_add = "QuantizeV2,Dequantize"

            graph_options = itex.GraphOptions(auto_mixed_precision_options=auto_mixed_precision_options)
            graph_options.auto_mixed_precision = itex.ON

            config = itex.ConfigProto(graph_options=graph_options)
            try:
                itex.set_backend("cpu", config)
            except TypeError:
                itex.set_config(config)
            config.graph_options.rewrite_options.constant_folding = rewriter_config_pb2.RewriterConfig.OFF

        sess = tf.compat.v1.Session(graph=graph, config=config)
        if (self.args.accuracy_only):
            print("Inference with real data")
            preprocessed_data_dir = "build/preprocessed_data"
            with open(os.path.join(preprocessed_data_dir, "preprocessed_files.pkl"), "rb") as f:
                preprocessed_files = pickle.load(f)

            dictionaries = []
            for preprocessed_file in preprocessed_files:
                with open(os.path.join(preprocessed_data_dir, preprocessed_file + ".pkl"), "rb") as f:
                    dct = pickle.load(f)[1]
                    dictionaries.append(dct)

            count = len(preprocessed_files)
            predictions = [None] * count
            validation_indices = list(range(0,count))
            print("Found {:d} preprocessed files".format(count))
            loaded_files = {}
            batch_size = self.args.batch_size
            # Get the number of steps based on batch size
            steps = count#math.ceil(count/batch_size)
            for i in range(steps):
                print("Iteration {} ...".format(i))
                test_data_index = validation_indices[i]#validation_indices[i * batch_size:(i + 1) * batch_size]
                file_name = preprocessed_files[test_data_index]
                with open(os.path.join(preprocessed_data_dir, "{:}.pkl".format(file_name)), "rb") as f:
                    data = pickle.load(f)[0]
                predictions[i] = sess.run(output_tensor, feed_dict={input_tensor: data[np.newaxis, ...]})[0].astype(np.float32)

            output_folder = "build/postprocessed_data"
            output_files = preprocessed_files
            # Post Process
            postprocess_output(predictions, dictionaries, validation_indices, output_folder, output_files)

            ground_truths = "build/raw_data/nnUNet_raw_data/Task043_BraTS2019/labelsTr"
            # Run evaluation
            print("Running evaluation...")
            evaluate_regions(output_folder, ground_truths, get_brats_regions())
            # Load evaluation summary
            print("Loading evaluation summary...")
            with open(os.path.join(output_folder, "summary.csv")) as f:
                for line in f:
                    words = line.split(",")
                    if words[0] == "mean":
                        whole = float(words[1])
                        core = float(words[2])
                        enhancing = float(words[3])
                        mean = (whole + core + enhancing) / 3
                        print("Accuracy: mean = {:.5f}, whole tumor = {:.4f}, tumor core = {:.4f}, enhancing tumor = {:.4f}".format(mean, whole, core, enhancing))
                        break

        print("Done!")

if __name__ == "__main__":
    evaluate_opt_graph = unet_3d_tf()
    evaluate_opt_graph.run()

