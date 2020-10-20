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
# SPDX-License-Identifier: EPL-2.0
#

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image
import time
import argparse
from tensorflow.python.client import timeline


parser = argparse.ArgumentParser()
parser.add_argument('-g', '--graph', help='Path to input graph to run', type=str, required=True)
parser.add_argument('-d', '--dataset', help='Full Path to input dataset to run', type=str, required=True)
parser.add_argument('-s', '--single_image', help='Run for single image onle, if false, run for the whole dataset', action='store_true')
parser.add_argument('-x', '--single_socket', help='Run for single socket, if false, run both sockets', action='store_true')
parser.add_argument('-v', '--visualize', help='Whether to visulize the output image', action='store_true' )
parser.add_argument('-t', '--timeline', help='Output file name for TF timeline', type=str, default=None)
parser.add_argument('-e', '--evaluate_tensor', help='Full tensor name to evaluate', type=str, default=None)
parser.add_argument('-p', '--print_accuracy', help='Print accuracy results', action='store_true')
parser.add_argument('-n', '--number_of_steps', help='Run for n number of steps', type=int, default=None)
parser.add_argument('--num-inter-threads', help='Num inter threads', type=int, default=None, dest="num_inter_threads")
parser.add_argument('--num-intra-threads', help='Num intra threads', type=int, default=None, dest="num_intra_threads")

args = parser.parse_args()


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# This is needed to display the images.
if (args.visualize and args.single_image and args.evaluate_tensor is None):
  from IPython import get_ipython
  get_ipython().run_line_magic('matplotlib', 'tk')

import importlib
label_map_util = importlib.import_module('..label_map_util', package='object_detection.utils.label_map_util')
vis_util = importlib.import_module('..visualization_utils', package='object_detection.utils.visualization_utils')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('object_detection/data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(args.graph, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  if image.mode == 'L':
    np_image = np.array(image.getdata()).reshape(
        (im_height, im_width)).astype(np.uint8)
    return np.stack((np_image,)*3, -1)
  else:
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

if (args.single_image):
  TEST_IMAGE_PATHS = [args.dataset + "/000000578871.jpg"]  
else:
  PATH_TO_TEST_IMAGES_DIR = args.dataset 
  print(PATH_TO_TEST_IMAGES_DIR)
  TEST_IMAGE_PATHS = []
  for root, dirs, files in os.walk(PATH_TO_TEST_IMAGES_DIR):
    for file in files:
      TEST_IMAGE_PATHS.append(os.path.join(PATH_TO_TEST_IMAGES_DIR, file))

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


def run_inference_for_single_image(graph):
  sess_config = tf.ConfigProto()
  sess_config.intra_op_parallelism_threads = args.num_intra_threads
  sess_config.inter_op_parallelism_threads = args.num_inter_threads
  if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = args.num_intra_threads

  with graph.as_default():
    with tf.Session(config=sess_config) as sess:
      # Get handles to input and output tensors
      tensor_dict = {}
      if (args.evaluate_tensor is None):
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes' 
        ]:
          tensor_name = key + ':0'
          if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                tensor_name)
      else:
        our_op = tf.get_default_graph().get_operation_by_name(args.evaluate_tensor)
        tensor_names = our_op.outputs
        list_ops = []
        for i, tensor in enumerate(tensor_names):
          list_ops.append(tensor.name)
        tensor_dict[args.evaluate_tensor] = list_ops

      if (args.timeline is not None):
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
      total_duration = 0
      for index, image_path in enumerate(TEST_IMAGE_PATHS):
        image = Image.open(image_path)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image)
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

        # Run inference
        start_time = time.time()
        if (args.timeline is not None):
          output_dict = sess.run(tensor_dict,
                                feed_dict={image_tensor: np.expand_dims(image_np, 0)}, options=run_options, run_metadata=run_metadata)
        else:
          output_dict = sess.run(tensor_dict,
                                 feed_dict={image_tensor: np.expand_dims(image_np, 0)})
        step_duration = time.time() - start_time
        if(index > 20):
          total_duration = total_duration + step_duration

        if (args.single_image):
          if index == 0:
            print('Avg. Duration per Step:' + str(total_duration / 1))
        else:
          if (index % 10 == 0):
            print('Step ' + str(index) + ': ' + str(step_duration) + ' seconds')
          if index == 4999:
            print('Avg. Duration per Step:' + str(total_duration / 5000))

        # Flush print messages
        sys.stdout.flush()

        if (args.number_of_steps is not None):
          if (args.single_image):
            sys.exit("single_iamge and number_of_steps cannot be both enabled!")
          elif (index == (args.number_of_steps - 1)):
            print('Avg. Duration per Step:' +
                  str(total_duration / (args.number_of_steps - 20)))
            break

        if (args.timeline is not None):
          trace = timeline.Timeline(step_stats=run_metadata.step_stats)
          with open('tl-' + time.strftime("%Y%m%d-%H%M%S") + '-' + args.timeline, 'w') as file:
            file.write(trace.generate_chrome_trace_format(show_memory=False))


        if (args.evaluate_tensor is not None):
          for tensor in output_dict[args.evaluate_tensor]:
            print(tensor.shape)
          return None, None

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict[
            'detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]

        if (args.print_accuracy):
          print('num_detections:\n' + str(output_dict['num_detections']))
          print('detection_classes:\n' + str(output_dict['detection_classes']))
          print('detection_boxes:\n' + str(output_dict['detection_boxes']))
          print('detection_scores:\n' + str(output_dict['detection_scores']))

        if 'detection_masks' in output_dict:
          output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict, image_np


# Actual detection.
output_dict, image_np = run_inference_for_single_image(detection_graph)

# Visualization of the results of a detection.
if (args.visualize and args.single_image and args.evaluate_tensor is None):
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=8)
  plt.figure(figsize=IMAGE_SIZE)
  plt.imshow(image_np)
