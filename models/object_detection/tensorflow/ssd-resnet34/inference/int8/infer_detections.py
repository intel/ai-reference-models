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

import os

import sys
import numpy as np
import cv2
import numpy
import json
import math
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from PIL import Image
import coco_constants

from tensorflow.core.protobuf import rewriter_config_pb2

class MyEncoder(json.JSONEncoder):
   def default(self, obj):
     if isinstance(obj, (numpy.int_, numpy.intc, numpy.intp, numpy.int8,
       numpy.int16, numpy.int32, numpy.int64, numpy.uint8,
       numpy.uint16,numpy.uint32, numpy.uint64)):
       return int(obj)
     elif isinstance(obj, (numpy.float_, numpy.float16, numpy.float32, 
       numpy.float64)):
       return float(obj)
     elif isinstance(obj, (numpy.ndarray,)): 
       return obj.tolist() 
     return json.JSONEncoder.default(self, obj)

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

    arg_parser.add_argument("--input-size",
                            help="Input image size, 300 or 1200",
                            dest="input_size", type=int, default=300)

    arg_parser.add_argument("--warmup-steps",
                            help="Number of warmup steps",
                            dest='warmup_steps', type=int, default=200)

    arg_parser.add_argument("--steps",
                            help="Number of steps",
                            dest='steps', type=int, default=800)

    arg_parser.add_argument('--onednn-graph', dest='onednn_graph',
                            help='enable OneDNN Graph', action='store_true')

    # parse the arguments
    self.args = arg_parser.parse_args()

    self.freeze_graph = self.load_graph(self.args.input_graph)
    self.config = tf.compat.v1.ConfigProto()
    self.config.intra_op_parallelism_threads = self.args.num_intra_threads
    self.config.inter_op_parallelism_threads = self.args.num_inter_threads
    if self.args.onednn_graph:
      self.config.graph_options.rewrite_options.constant_folding = rewriter_config_pb2.RewriterConfig.OFF


    if self.args.batch_size == -1:
      self.args.batch_size = 64
    
    self.num_batches = (ssd_constants.COCO_NUM_VAL_IMAGES // self.args.batch_size) + \
                       (ssd_constants.COCO_NUM_VAL_IMAGES % self.args.batch_size > 0)
  
    if self.args.input_size == 300:
      input_layer = 'input'
      output_layers = ['v/stack', 'v/Softmax']
      self.input_tensor = self.freeze_graph.get_tensor_by_name(input_layer + ":0")
      self.output_tensors = [self.freeze_graph.get_tensor_by_name(x + ":0") for x in output_layers]
    elif self.args.input_size == 1200:
      with self.freeze_graph.as_default():
        self.input_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('image:0')
        self.output_tensors = {}
        for key in ['detection_bboxes', 'detection_scores','detection_classes']:
          tensor_name = key + ':0'
          self.output_tensors[key] =tf.compat.v1.get_default_graph().get_tensor_by_name(
                tensor_name)


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
      if self.args.input_size == 300 or self.args.input_size == 1200:
        input_images = sess.run(tf.random.truncated_normal(
            [self.args.batch_size, self.args.input_size, self.args.input_size, 3],
            dtype=tf.float32,
            stddev=10,
            name='synthetic_images'))
      else:
        raise Exception('input size unsupported')

      total_iter = self.args.steps + self.args.warmup_steps
      warmup_iter = self.args.warmup_steps
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
      output_shapes=[[self.args.batch_size, self.args.input_size, self.args.input_size, 3]],
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


  def accuracy_check_300(self):
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
  
  def preprocess(self, image): 
    image = image.astype('float32') / 255.
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    image = (image - mean) / std
    return image


  def load_batch(self, niter, image_lines, image_root, batch_size): 
    if (niter - 1) * batch_size >= len(image_lines):
      raise Exception("%d batch exceed the number of images!"%niter)
    image_batch = []
    image_name_batch = []
    image_shape_batch = []
    image_index = (niter - 1) * batch_size
    while image_index < len(image_lines) and image_index < niter * batch_size:
      image_name = image_lines[image_index].strip() 
      image_path = os.path.join(image_root, image_name + ".jpg")

      image = Image.open(image_path).convert("RGB")
      width, height = image.size
      image=np.array(image.resize((self.args.input_size,self.args.input_size), Image.BILINEAR))
      image = self.preprocess(image)
      image_batch.append(image)
      image_name_batch.append(image_name)
      image_shape_batch.append([height, width])
      image_index += 1
    return np.stack(image_batch, axis=0), image_name_batch, image_shape_batch  

  def ssd_parse_proto(self, serialized):
    feature_map = {
        'image/encoded': tf.io.FixedLenFeature(
            (), dtype=tf.string, default_value=''),
        'image/source_id': tf.io.FixedLenFeature((), tf.string, default_value=''),
    }
    features = tf.io.parse_single_example(serialized=serialized, features=feature_map)

    image_buffer = features['image/encoded']
    image = tf.image.decode_jpeg(image_buffer)
    source_id = features['image/source_id']

    return image, source_id

  def load_batch_tfrecord(self, niter, data_group, batch_size, sess): 
    if (niter - 1) * batch_size >= ssd_constants.COCO_NUM_VAL_IMAGES :
      raise Exception("%d batch exceed the number of images!"%niter)
    image_batch = []
    image_name_batch = []
    image_shape_batch = []
    image_index = (niter - 1) * batch_size
    while image_index < ssd_constants.COCO_NUM_VAL_IMAGES  and image_index < niter * batch_size:
      image, source_id = sess.run(data_group)

      image_name = int(source_id)
      height, width = image.shape[0:2]
      if image.shape[2] == 1:
        image = np.concatenate((image, image, image), axis=2)
      image = cv2.resize(image, (self.args.input_size, self.args.input_size))
      image = self.preprocess(image)
      
      image_batch.append(image)
      image_name_batch.append(image_name)
      image_shape_batch.append([height, width])
      image_index += 1
    return np.stack(image_batch, axis=0), image_name_batch, image_shape_batch  


  def cocoval(self, detected_json, eval_json):
    eval_gt = COCO(eval_json)
    eval_dt = eval_gt.loadRes(detected_json)
    cocoEval = COCOeval(eval_gt, eval_dt, iouType='bbox')

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize() 

  def run_inference_for_eval(self, graph):
    with graph.as_default():
      with tf.compat.v1.Session(config=self.config) as sess:

        num_iter = self.num_batches

        coco_records = []
        count = 0
        labelmap = coco_constants.LABEL_MAP

        tfrecord_path = os.path.join(self.args.data_location, "validation-00000-of-00001")
        dataset = tf.compat.v1.data.TFRecordDataset(tfrecord_path)
        dataset = dataset.map(self.ssd_parse_proto)
        dataset = dataset.prefetch(self.args.batch_size)
        iterator = dataset.make_one_shot_iterator()
        data_group = iterator.get_next()

        for ind in range(num_iter):         
          image_batch, image_name_batch, image_shape_batch = self.load_batch_tfrecord(ind + 1, data_group, self.args.batch_size, sess)  
          count += image_batch.shape[0] 
          print("process: %d images"%count)
          output_dict = sess.run(self.output_tensors,
                                feed_dict={self.input_tensor: image_batch})
          cur_batch_size = image_batch.shape[0] 
          for bind in range(cur_batch_size):
            num_detections = int(output_dict['detection_bboxes'][bind].shape[0])
            for ind_bb in range(num_detections):
              record = {}
              height, width = image_shape_batch[bind]
              ymin = output_dict['detection_bboxes'][bind][ind_bb][0] * height
              xmin = output_dict['detection_bboxes'][bind][ind_bb][1] * width
              ymax = output_dict['detection_bboxes'][bind][ind_bb][2] * height
              xmax = output_dict['detection_bboxes'][bind][ind_bb][3] * width
              score = output_dict['detection_scores'][bind][ind_bb]
              class_id = int(output_dict['detection_classes'][bind][ind_bb])
              record['image_id'] = int(image_name_batch[bind])
              record['category_id'] = labelmap[class_id]
              record['score'] = score
              record['bbox'] = [xmin, ymin, xmax - xmin + 1, ymax - ymin + 1]
              if score < coco_constants.SCORE_THRESHOLD:
                break 
              coco_records.append(record)
    return coco_records  
  
  def accuracy_check_1200(self):
    detection_graph = self.freeze_graph 
    coco_records = self.run_inference_for_eval(detection_graph)
    det_file = "/tmp/ssd_resnet_coco_det.json"
    with open(det_file, 'w') as f_det:
      f_det.write(json.dumps(coco_records, cls=MyEncoder))
    gt_path = os.path.join(self.args.data_location, 'annotations', 'instances_val2017.json')
    self.cocoval(det_file, gt_path)



  def run(self):
    if self.args.accuracy_only:
      if self.args.input_size == 300:
        self.accuracy_check_300()
      elif self.args.input_size == 1200:
        self.accuracy_check_1200()
    else:
      self.run_benchmark()

if __name__ == "__main__":
  infer = ssd_resnet34_infer()
  infer.run()

