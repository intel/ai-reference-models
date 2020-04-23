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

import tensorflow as tf
import time

from argparse import ArgumentParser

import benchmark_cnn
import datasets
import ssd_constants
from models import ssd_model
from preprocessing import COCOPreprocessor

IMAGE_SIZE = 300

import os

class ssd_resnet34_infer:

  def __init__(self):
    arg_parser = ArgumentParser(description='Parse args')

    arg_parser.add_argument('-b', "--batch-size",
                            help="Specify the batch size. If this " \
                                 "parameter is not specified or is -1, the " \
                                 "largest ideal batch size for the model will " \
                                 "be used.",
                            dest="batch_size", type=int, default=-1)

    arg_parser.add_argument('-e', "--inter-op-parallelism-threads",
                            help='The number of inter-thread.',
                            dest='num_inter_threads', type=int, default=0)

    arg_parser.add_argument('-a', "--intra-op-parallelism-threads",
                            help='The number of intra-thread.',
                            dest='num_intra_threads', type=int, default=0)

    arg_parser.add_argument('-g', "--input-graph",
                            help='Specify the input graph.',
                            dest='input_graph')

    arg_parser.add_argument('-d', "--data-location",
                            help='Specify the location of the data. '
                                 'If this parameter is not specified, '
                                 'the benchmark will use random/dummy data.',
                            dest="data_location", default=None)

    arg_parser.add_argument('-r', "--accuracy-only",
                            help='For accuracy measurement only.',
                            dest='accuracy_only', action='store_true')

    arg_parser.add_argument("--results-file-path",
                            help="File path for the inference results",
                            dest="results_file_path", default=None)

    # parse the arguments
    self.args = arg_parser.parse_args()

    self.freeze_graph = self.load_graph(self.args.input_graph)
    self.config = tf.compat.v1.ConfigProto()
    self.config.intra_op_parallelism_threads = self.args.num_intra_threads
    self.config.inter_op_parallelism_threads = self.args.num_inter_threads

    if self.args.batch_size == -1:
      self.args.batch_size = 64

    self.num_batches = (ssd_constants.COCO_NUM_VAL_IMAGES // self.args.batch_size) + \
                       (ssd_constants.COCO_NUM_VAL_IMAGES % self.args.batch_size > 0)
    
    input_layer = 'input'
    output_layers = ['v/stack', 'v/Softmax']
    self.input_tensor = self.freeze_graph.get_tensor_by_name(input_layer + ":0")
    self.output_tensors = [self.freeze_graph.get_tensor_by_name(x + ":0") for x in output_layers]
    

  def load_graph(self, frozen_graph_filename):
    print('load graph from: ' + frozen_graph_filename)
    with tf.io.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name='')
    return graph

  def run_benchmark(self):
    print("Inference with dummy data.")
    with tf.compat.v1.Session(graph=self.freeze_graph, config=self.config) as sess:
      
      input_images = sess.run(tf.random.truncated_normal(
          [self.args.batch_size, IMAGE_SIZE, IMAGE_SIZE, 3],
          dtype=tf.float32,
          stddev=10,
          name='synthetic_images'))

      total_iter = 1000
      warmup_iter = 200
      ttime = 0.0

      print('total iteration is {0}'.format(str(total_iter)))
      print('warm up iteration is {0}'.format(str(warmup_iter)))

      for step in range(total_iter):
        start_time = time.time()
        _ = sess.run(self.output_tensors, {self.input_tensor: input_images})
        end_time = time.time()

        duration = end_time - start_time
        if (step + 1) % 10 == 0:
          print('steps = {0}, {1} sec'.format(str(step), str(duration)))
        
        if step + 1 > warmup_iter:
          ttime += duration
        
      total_batches = total_iter - warmup_iter
      print ('Batchsize: {0}'.format(str(self.args.batch_size)))
      print ('Time spent per BATCH: {0:10.4f} ms'.format(ttime / total_batches * 1000))
      print ('Total samples/sec: {0:10.4f} samples/s'.format(total_batches * self.args.batch_size / ttime))
  

  def __get_input(self):
    preprocessor = COCOPreprocessor(
      batch_size=self.args.batch_size,
      output_shapes=[[self.args.batch_size, IMAGE_SIZE, IMAGE_SIZE, 3]],
      num_splits=1,
      dtype=tf.float32,
      train=False,
      distortions=True,
      resize_method=None,
      shift_ratio=0
    )

    class params:
      datasets_repeat_cached_sample = False

    self.params = params()
    self.dataset = datasets.create_dataset(self.args.data_location, 'coco')
    
    return preprocessor.minibatch(
      self.dataset,
      subset='validation',
      params=self.params,
      shift_ratio=0)


  def accuracy_check(self):
    print(self.args)
    input_list = self.__get_input()
    ds_init = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TABLE_INITIALIZERS)

    ds_sess = tf.compat.v1.Session()
    params = benchmark_cnn.make_params(data_dir=self.args.data_location)
    self.model = ssd_model.SSD300Model(params=params)

    print("Inference for accuracy check.")
    with tf.compat.v1.Session(graph=self.freeze_graph, config=self.config) as sess:
      ds_sess.run(ds_init)
      global_step = 0

      for _ in range(self.num_batches):
        results = {}
        input_lists = ds_sess.run(input_list)
        input_images = input_lists[0][0]
        input_ids = input_lists[3][0]
        input_raw_shapes = input_lists[4][0]

        result = sess.run(self.output_tensors, {self.input_tensor: input_images})
        # Make global_step available in results for postprocessing.
        results['global_step'] = global_step
        results[ssd_constants.SOURCE_ID] = input_ids
        results[ssd_constants.RAW_SHAPE] = input_raw_shapes

        results[ssd_constants.PRED_BOXES] = result[0]
        results[ssd_constants.PRED_SCORES] = result[1]

        results = self.model.postprocess(results)



  def run(self):
    if self.args.accuracy_only:
      self.accuracy_check()
    else:
      self.run_benchmark()



if __name__ == "__main__":
  infer = ssd_resnet34_infer()
  infer.run()

