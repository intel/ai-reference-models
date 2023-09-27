#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Intel Corporation
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

import time
from argparse import ArgumentParser

import tensorflow as tf
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
from tensorflow.python.framework import dtypes
from tensorflow.core.protobuf import rewriter_config_pb2

import numpy as np
import os
import threading

INPUTS = 'input_tensor'
OUTPUTS = 'softmax_tensor'

RESNET_IMAGE_SIZE = 224


class eval_classifier_optimized_graph:
  """Evaluate image classifier with optimized TensorFlow graph"""

  def __init__(self):

    arg_parser = ArgumentParser(description='Parse args')

    arg_parser.add_argument('-b', "--batch-size",
                            help="Specify the batch size. If this " \
                                 "parameter is not specified or is -1, the " \
                                 "largest ideal batch size for the model will " \
                                 "be used.",
                            dest="batch_size", type=int, default=-1)

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

    arg_parser.add_argument('-d', "--data-location",
                            help='Specify the location of the data. '
                                 'If this parameter is not specified, '
                                 'the benchmark will use random/dummy data.',
                            dest="data_location", default=None)

    arg_parser.add_argument('-r', "--accuracy-only",
                            help='For accuracy measurement only.',
                            dest='accuracy_only', action='store_true')
    arg_parser.add_argument('--calibrate', dest='calibrate',
                            help='Run accuracy with calibration data,'
                                 'to generate min_max ranges, calibrate=[True/False]',
                            type=bool, default=False)
    arg_parser.add_argument("--results-file-path",
                            help="File path for the inference results",
                            dest="results_file_path", default=None)
    arg_parser.add_argument("--warmup-steps", type=int, default=10,
                            help="number of warmup steps")
    arg_parser.add_argument("--steps", type=int, default=50,
                            help="number of steps")

    arg_parser.add_argument(
      '--data-num-inter-threads', dest='data_num_inter_threads',
      help='number threads across operators',
      type=int, default=32)
    arg_parser.add_argument(
      '--data-num-intra-threads', dest='data_num_intra_threads',
      help='number threads for data layer operator',
      type=int, default=14)
    arg_parser.add_argument(
      '--num-cores', dest='num_cores',
      help='number of cores',
      type=int, default=28)
    arg_parser.add_argument(
      '--onednn-graph', dest='onednn_graph',
      help='enable OneDNN Graph',
      action='store_true')

    self.args = arg_parser.parse_args()
    # validate the arguements
    self.validate_args()

  def write_results_output(self, predictions, filenames, labels):
    # If a results_file_path is provided, write the predictions to the file
    if self.args.results_file_path:
      top_predictions = np.argmax(predictions, 1)
      with open(self.args.results_file_path, "a") as fp:
        for filename, expected_label, top_prediction in zip(filenames, labels, top_predictions):
          fp.write("{},{},{}\n".format(filename, expected_label, top_prediction))

  def run(self):
    """run benchmark with optimized graph"""

    print("Run inference")

    data_config = tf.compat.v1.ConfigProto()
    data_config.intra_op_parallelism_threads = self.args.data_num_intra_threads
    data_config.inter_op_parallelism_threads = self.args.data_num_inter_threads

    infer_config = tf.compat.v1.ConfigProto()
    infer_config.intra_op_parallelism_threads = self.args.num_intra_threads
    infer_config.inter_op_parallelism_threads = self.args.num_inter_threads
    infer_config.graph_options.rewrite_options.remapping = (
                  rewriter_config_pb2.RewriterConfig.AGGRESSIVE)
    if self.args.onednn_graph:
      infer_config.graph_options.rewrite_options.constant_folding = rewriter_config_pb2.RewriterConfig.OFF

    infer_graph = tf.Graph()

    with infer_graph.as_default():
      graph_def = tf.compat.v1.GraphDef()
      with tf.compat.v1.gfile.FastGFile(self.args.input_graph, 'rb') as input_file:
        input_graph_content = input_file.read()
        graph_def.ParseFromString(input_graph_content)

      output_graph = optimize_for_inference(graph_def, [INPUTS], 
                              [OUTPUTS], dtypes.float32.as_datatype_enum, False)
      tf.import_graph_def(output_graph, name='')

    # Definite input and output Tensors for detection_graph
    input_tensor = infer_graph.get_tensor_by_name('input_tensor:0')
    output_tensor = infer_graph.get_tensor_by_name('softmax_tensor:0')

    data_graph = tf.Graph()
    with data_graph.as_default():
      input_shape = [self.args.batch_size, RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE, 3]
      images = tf.random.uniform(input_shape, 0.0, 255.0, dtype=tf.float32, name='synthetic_images')

    infer_sess1 = tf.compat.v1.Session(graph=infer_graph, config=infer_config)
    data_sess  = tf.compat.v1.Session(graph=data_graph,  config=data_config)

    image_list, throughput_list = [], []
    def run_model(data_sess, infer_sess, tid):
        start_time = time.time()
        num_images = 0
        warm_up = self.args.warmup_steps
        steps = self.args.steps
        time_consume = 0
        image_np = data_sess.run(images)
        while num_images < steps:
          start_time = time.time()
          predictions = infer_sess.run(output_tensor, feed_dict={input_tensor: image_np})
          end_time = time.time()
          if (num_images > warm_up):
            time_consume += end_time - start_time
          num_images+=1
        if self.args.onednn_graph:
          total_examples = (steps - warm_up) * self.args.batch_size
          single_throughput = total_examples/time_consume
          throughput_list.append(single_throughput)
          print('Instance num %d Avg Time/Iteration %f msec Throughput %f images/sec' %(tid, time_consume*1000/total_examples, total_examples/time_consume))
        else:
          throughput_list.append((steps-warm_up)/time_consume)
          print('Instance num %d Avg Time/Iteration %f msec Throughput %f images/sec' %(tid, time_consume*1000/(steps - warm_up), (steps-warm_up)/time_consume))

    if (not self.args.accuracy_only):
      if self.args.onednn_graph and os.environ.get("_ITEX_ONEDNN_GRAPH_COMPILER_BACKEND")=="1":
        # graph compiler requires 1 core per instance
        num_instances = self.args.num_intra_threads
      else:
        num_instances = self.args.num_intra_threads//4
      threads = []
      for i in range(1, num_instances+1):
        thread = threading.Thread(target=run_model, args=(data_sess,infer_sess1, i))
        threads.append(thread)
        thread.start()

      for index, thread in enumerate(threads):
        thread.join()

      total_throughput = 0
      for i in range(0, num_instances):
         total_throughput += throughput_list[i]
      print('Total aggregated Throughput %f' %(total_throughput))

    else: # accuracy check
      total_accuracy1, total_accuracy5 = (0.0, 0.0)

      while num_remaining_images >= self.args.batch_size:
        # Reads and preprocess data
        tf_filenames = None
        if self.args.results_file_path:
          np_images, np_labels, tf_filenames = data_sess.run([images, labels, filenames])
        else:
          np_images, np_labels = data_sess.run([images, labels])
        num_processed_images += self.args.batch_size
        num_remaining_images -= self.args.batch_size

        start_time = time.time()
        # Compute inference on the preprocessed data
        predictions = infer_sess.run(output_tensor,
                               {input_tensor: np_images})
        elapsed_time = time.time() - start_time

        # Write out the file name, expected label, and top prediction
        self.write_results_output(predictions, tf_filenames, np_labels)

        with tf.Graph().as_default() as accu_graph:
          accuracy1 = tf.reduce_sum(
            input_tensor=tf.cast(tf.nn.in_top_k(predictions=tf.constant(predictions),
                                   targets=tf.constant(np_labels), k=1), tf.float32))

          accuracy5 = tf.reduce_sum(
            input_tensor=tf.cast(tf.nn.in_top_k(predictions=tf.constant(predictions),
                                   targets=tf.constant(np_labels), k=5), tf.float32))
          with tf.compat.v1.Session(config=infer_config) as accu_sess:
            np_accuracy1, np_accuracy5 = accu_sess.run([accuracy1, accuracy5])

          total_accuracy1 += np_accuracy1
          total_accuracy5 += np_accuracy5

        print("Iteration time: %0.4f ms" % elapsed_time)
        print("Processed %d images. (Top1 accuracy, Top5 accuracy) = (%0.4f, %0.4f)" \
                  % (num_processed_images, total_accuracy1 / num_processed_images,
                     total_accuracy5 / num_processed_images))

  def validate_args(self):
    """validate the arguments"""

    if not self.args.data_location:
      if self.args.accuracy_only:
        raise ValueError("You must use real data for accuracy measurement.")


if __name__ == "__main__":
  evaluate_opt_graph = eval_classifier_optimized_graph()
  evaluate_opt_graph.run()
