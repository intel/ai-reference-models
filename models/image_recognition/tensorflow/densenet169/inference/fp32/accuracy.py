# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import time
import numpy as np
from tensorflow.core.protobuf import rewriter_config_pb2
from google.protobuf import text_format
import tensorflow as tf
import image_preprocessing
import dataset

NUM_TEST_IMAGES = 50000

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.compat.v1.GraphDef()

  import os
  file_ext = os.path.splitext(model_file)[1]

  with open(model_file, "rb") as f:
    if file_ext == '.pbtxt':
      text_format.Merge(f.read(), graph_def)
    else:
      graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def, name='')
  return graph

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_graph", default=None,
                      help="graph/model to be executed")
  parser.add_argument("--data_location", default=None,
                      help="full path to the validation data")
  parser.add_argument("--input_height", default=224,
                      type=int, help="input height")
  parser.add_argument("--input_width", default=224,
                      type=int, help="input width")
  parser.add_argument("--batch_size", default=32,
                      type=int, help="batch size")
  parser.add_argument("--input_layer", default="input",
                      help="name of input layer")
  parser.add_argument("--output_layer", default="densenet169/predictions/Reshape_1",
                      help="name of output layer")
  parser.add_argument(
      '--num_inter_threads',
      help='number threads across operators',
      type=int, default=1)
  parser.add_argument(
      '--num_intra_threads',
      help='number threads for an operator',
      type=int, default=1)
  args = parser.parse_args()

  if args.input_graph:
    model_file = args.input_graph
  else:
    sys.exit("Please provide a graph file.")
  input_height = args.input_height
  input_width = args.input_width
  batch_size = args.batch_size
  input_layer = args.input_layer
  output_layer = args.output_layer
  num_inter_threads = args.num_inter_threads
  num_intra_threads = args.num_intra_threads
  data_location = args.data_location

  data_graph = tf.Graph() ###
  with data_graph.as_default(): ###
    dataset = dataset.ImagenetData(data_location)
    preprocessor = image_preprocessing.ImagePreprocessor(
        input_height, input_width, batch_size,
        1, # device count
        tf.float32, # data_type for input fed to the graph
        train=False, # doing inference
        resize_method='crop')
    images, labels = preprocessor.minibatch(dataset, subset='validation')

  graph = load_graph(model_file)
  input_tensor = graph.get_tensor_by_name(input_layer + ":0")
  output_tensor = graph.get_tensor_by_name(output_layer + ":0")

  rewrite_options = rewriter_config_pb2.RewriterConfig(
          layout_optimizer=rewriter_config_pb2.RewriterConfig.ON)

  config = tf.compat.v1.ConfigProto()
  config.inter_op_parallelism_threads = num_inter_threads
  config.intra_op_parallelism_threads = num_intra_threads

  config.graph_options.rewrite_options.remapping = (
          rewriter_config_pb2.RewriterConfig.OFF)

  total_accuracy1, total_accuracy5 = (0.0, 0.0)
  num_processed_images = 0
  num_remaining_images = dataset.num_examples_per_epoch(subset='validation') \
                            - num_processed_images
  top1 = 0
  with tf.compat.v1.Session(graph=data_graph) as sess: ###
    sess_graph = tf.compat.v1.Session(graph=graph, config=config)

    while num_remaining_images >= batch_size:
      # Reads and preprocess data
      #import pdb
      #pdb.set_trace()
      np_images, np_labels = sess.run([images[0], labels[0]])
      np_labels -= 1
      #print(np_labels.shape)
      num_processed_images += batch_size
      num_remaining_images -= batch_size
      start_time = time.time()
      # Compute inference on the preprocessed data
      predictions1 = sess_graph.run(output_tensor,
                             {input_tensor: np_images})
      elapsed_time = time.time() - start_time
      if(batch_size !=1):
         predictions1 = sess.run(tf.squeeze(predictions1))
      else :
         predictions1 = sess.run(tf.reshape(predictions1,[1,1000]))
      predictions2 = tf.argmax(input=predictions1, axis=1)
      predictions = sess.run(predictions2)
      top1 += batch_size - (np.count_nonzero(predictions - np_labels))
      print("Iteration time: %0.4f ms" % elapsed_time)
      print(top1/num_processed_images)
