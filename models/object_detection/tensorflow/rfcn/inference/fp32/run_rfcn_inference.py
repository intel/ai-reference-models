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

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import subprocess

from collections import defaultdict
from io import StringIO
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image
import time
import argparse
from tensorflow.python.client import timeline
import importlib

class RFCNRunner:
  '''Add code here to detect the environment and set necessary variables before launching the model'''
  args=None
  custom_args=None
  RESEARCH_DIR = 'research'
  OBJ_DETECTION_DIR = 'object_detection'
  DATA_DIR = 'data'
  label_map_file = 'mscoco_label_map.pbtxt'
  research_dir = ''
  NUM_CLASSES = 90
  detection_graph = None
  test_image_paths = []
  TEST_IMG_FILE = "000000578871.jpg"
  DEFAULT_INTEROP_THREADS = 2
  RFCN_OUTPUTS = ['num_detections', 'detection_boxes', 'detection_scores',
              'detection_classes', 'detection_masks']
  STEP_SIZE = 10
  MAX_STEPS = 5000
  label_map_util = None
  vis_util = None
  # Size, in inches, of the output images.
  IMAGE_SIZE = (12, 8)

  def __init__(self, args):
    self.args = args
    self.parse_args()
    self.log('Received these standard args: {}'.format(self.args))

  def parse_args(self):
    parser = argparse.ArgumentParser()
    mutex_group = parser.add_mutually_exclusive_group()
    mutex_group.add_argument('-x', '--number_of_steps', help='Run for n number of steps', type=int, default=None)
    mutex_group.add_argument('-z', '--visualize', help='Whether to visulize the output image', action='store_true' )
    parser.add_argument('-v', '--verbose', help='Print some useful info.', action='store_true' )
    parser.add_argument('-t', '--timeline', help='Output file name for TF timeline', type=str, default=None)
    parser.add_argument('-e', '--evaluate_tensor', help='Full tensor name to evaluate', type=str, default=None)
    parser.add_argument('-p', '--print_accuracy', help='Print accuracy results', action='store_true')
    parser.add_argument('-g', '--input_graph', help='The input frozen graph pb file', dest='input_graph', required=True, default=None)
    parser.add_argument('-d', '--data_location', help='The location of the image data to be analyzed.', dest='data_location', default=None, required=True)
    parser.add_argument('-m', '--tensorflow-models-path',
        help='Path to the tensorflow-models directory (or clone of github.com/tensorflow/models',
        dest='tf_models_path', default=None, required=True)
    parser.add_argument(
        '--num-inter-threads', dest='num_inter_threads',
        help='number threads across operators',
        type=int, default=2)
    parser.add_argument(
        '--num-intra-threads', dest='num_intra_threads',
        help='number threads for an operator',
        type=int, default=56)
    self.args = parser.parse_args()
    self.validate_args()
    self.finish_import()

  def log(self, msg):
    if self.args.verbose: print(msg)

  def validate_args(self):
    self.log('Validating Args...')
    self.research_dir = os.path.join(self.args.tf_models_path, self.RESEARCH_DIR)
    if not ( self.args.data_location and
        os.path.exists(os.path.join(self.args.data_location, self.TEST_IMG_FILE))):
      raise ValueError ("Unable to locate images for evaluation at {}".format(self.args.data_location))
    if os.path.isdir(self.research_dir):
      # List of the strings that is used to add correct label for each box.
      self.label_map_file = os.path.join(self.research_dir,
                                          self.OBJ_DETECTION_DIR,
                                          self.DATA_DIR,
                                          self.label_map_file)
      if not os.path.exists(self.label_map_file):
        raise ValueError ("Unable to locate label map file at {}".format(self.label_map_file))
    else:
      raise ValueError ("{} is not a valid path to the TensorFlow models.".format(self.args.tf_models_path))

    if not os.path.exists(self.args.input_graph):
      raise ValueError("Unable to find the input graph protobuf file: {}".format(self.args.input_graph))

  def finish_import(self):
    # This is needed since the notebook is stored in the object_detection folder.
    sys.path.append(self.research_dir)
    # This is needed to display the images.
    if (self.args.visualize and self.args.evaluate_tensor is None):
      from IPython import get_ipython
      get_ipython().run_line_magic('matplotlib', 'tk')
    self.label_map_util = importlib.import_module('..label_map_util', package='object_detection.utils.label_map_util')
    self.vis_util = importlib.import_module('..visualization_utils', package='object_detection.utils.visualization_utils')

  def run(self):
      self.log("Running performance test")
      self.read_graph()
      self.get_image_paths()
      #self.load_label_map()
      # Actual detection.
      output_dict, image_np = self.run_inference(self.detection_graph)
      self.visualize(output_dict, image_np)

  def visualize(self, output_dict, image_np):
    # Visualization of the results of a detection.
    if (self.args.visualize and
        self.args.evaluate_tensor is None and
        self.category_index and
        output_dict and
        image_np ):
      self.vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          output_dict['detection_boxes'],
          output_dict['detection_classes'],
          output_dict['detection_scores'],
          self.category_index,
          instance_masks=output_dict.get('detection_masks'),
          use_normalized_coordinates=True,
          line_thickness=8)
      plt.figure(figsize=self.IMAGE_SIZE)
      plt.imshow(image_np)

  def read_graph(self):
    self.detection_graph = tf.Graph()
    with self.detection_graph.as_default():
      od_graph_def = tf.compat.v1.GraphDef()
      with tf.io.gfile.GFile(self.args.input_graph, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


  def get_image_paths(self):
    if (self.args.visualize):
      self.test_image_paths = [os.path.join(self.args.data_location, self.TEST_IMG_FILE)]
    else:
      self.test_image_paths = []
      for root, dirs, files in os.walk(self.args.data_location):
        for file in files:
          self.test_image_paths.append(os.path.join(self.args.data_location, file))

  def load_label_map(self):
    label_map = self.label_map_util.load_labelmap(self.label_map_file)
    categories = self.label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=self.NUM_CLASSES, use_display_name=True)
    if (self.args.visualize and self.args.evaluate_tensor is None):
      self.category_index = self.label_map_util.create_category_index(categories)

  def load_image_into_numpy_array(self, image):
    (im_width, im_height) = image.size
    if image.mode == 'L':
      np_image = np.array(image.getdata()).reshape(
          (im_height, im_width)).astype(np.uint8)
      return np.stack((np_image,)*3, -1)
    else:
      return np.array(image.getdata()).reshape(
          (im_height, im_width, 3)).astype(np.uint8)

  def run_inference(self,graph):
    sess_config = tf.compat.v1.ConfigProto()
    sess_config.intra_op_parallelism_threads = self.args.num_intra_threads
    sess_config.inter_op_parallelism_threads = self.args.num_inter_threads
    with self.detection_graph.as_default():
      with tf.compat.v1.Session(config=sess_config) as sess:
        # Get handles to input and output tensors
        tensor_dict = {}
        if not self.args.evaluate_tensor:
          ops = tf.compat.v1.get_default_graph().get_operations()
          all_tensor_names = {output.name for op in ops for output in op.outputs}
          for key in self.RFCN_OUTPUTS:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
              tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(
                  tensor_name)
        else:
          our_op = tf.compat.v1.get_default_graph().get_operation_by_name(self.args.evaluate_tensor)
          tensor_names = our_op.outputs
          list_ops = []
          for i, tensor in enumerate(tensor_names):
            list_ops.append(tensor.name)
          tensor_dict[self.args.evaluate_tensor] = list_ops

        run_options = None
        run_metadata = None
        if self.args.timeline:
          run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
          run_metadata = tf.compat.v1.RunMetadata()

        total_duration = 0
        for index, image_path in enumerate(self.test_image_paths):
          image = Image.open(image_path)
          # the array based representation of the image will be used later in order to prepare the
          # result image with boxes and labels on it.
          image_np = self.load_image_into_numpy_array(image)
          image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('image_tensor:0')

          # Run inference
          start_time = time.time()
          #if self.args.timeline:
          output_dict = sess.run(tensor_dict,
                       feed_dict={image_tensor: np.expand_dims(image_np, 0)},
                       options=run_options, run_metadata=run_metadata)
          # else:
          #   output_dict = sess.run(tensor_dict,
          #                         feed_dict={image_tensor: np.expand_dims(image_np, 0)})
          step_duration = time.time() - start_time
          total_duration = total_duration + step_duration

          if (self.args.visualize):
            if index == 0:
              print ('Avg. Duration per Step:' + str(total_duration / 1))
          else:
            if (index % self.STEP_SIZE == 0):
              print ('Step ' + str(index) + ': ' + str(step_duration) + ' seconds', flush=True)
            if index == self.MAX_STEPS - 1:
              print ('Avg. Duration per Step:' + str(total_duration / self.MAX_STEPS))

          if self.args.number_of_steps and index == (self.args.number_of_steps - 1):
              print ('Avg. Duration per Step:' +
                    str(total_duration / self.args.number_of_steps))
              break

          if self.args.timeline:
            trace = timeline.Timeline(step_stats=run_metadata.step_stats)
            with open('tl-' + time.strftime("%Y%m%d-%H%M%S") + '-' + self.args.timeline, 'w') as file:
              file.write(trace.generate_chrome_trace_format(show_memory=False))

          if self.args.evaluate_tensor:
            for tensor in output_dict[self.args.evaluate_tensor]:
              print (tensor.shape)
            return None, None

          # all outputs are float32 numpy arrays, so convert types as appropriate
          output_dict['num_detections'] = int(output_dict['num_detections'][0])
          output_dict['detection_classes'] = output_dict[
              'detection_classes'][0].astype(np.uint8)
          output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
          output_dict['detection_scores'] = output_dict['detection_scores'][0]

          if (self.args.print_accuracy):
            print ('num_detections:\n' + str(output_dict['num_detections']))
            print ('detection_classes:\n' + str(output_dict['detection_classes']))
            print ('detection_boxes:\n' + str(output_dict['detection_boxes']))
            print ('detection_scores:\n' + str(output_dict['detection_scores']))

          if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict, image_np

if __name__ == "__main__":
  rr = RFCNRunner(sys.argv)
  rr.run()
