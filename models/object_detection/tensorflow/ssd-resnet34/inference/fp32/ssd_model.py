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

# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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


"""SSD300 Model Configuration.

References:
  Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed,
  Cheng-Yang Fu, Alexander C. Berg
  SSD: Single Shot MultiBox Detector
  arXiv:1512.02325

Ported from MLPerf reference implementation:
  https://github.com/mlperf/reference/tree/ssd/single_stage_detector/ssd

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import os
import re
import threading
import tensorflow as tf

import ssd_constants


class SSD300Model():
  """Single Shot Multibox Detection (SSD) model for 300x300 image datasets."""

  def __init__(self, data_dir, label_num=ssd_constants.NUM_CLASSES):
    # For COCO dataset, 80 categories + 1 background = 81 labels
    self.label_num = label_num
    self.data_dir = data_dir

    # Collected predictions for eval stage. It maps each image id in eval
    # dataset to a dict containing the following information:
    #   source_id: raw ID of image
    #   raw_shape: raw shape of image
    #   pred_box: encoded box coordinates of prediction
    #   pred_scores: scores of classes in prediction
    self.predictions = {}

    # Global step when predictions are collected.
    self.eval_global_step = 0

    # Average precision. In asynchronous eval mode, this is the latest AP we
    # get so far and may not be the results at current eval step.
    self.eval_coco_ap = 0

    # Process, queues, and thread for asynchronous evaluation. When enabled,
    # create a separte process (async_eval_process) that continously pull
    # intermediate results from the predictions queue (a multiprocessing queue),
    # process them, and push final results into results queue (another
    # multiprocessing queue). The main thread is responsible to push message
    # into predictions queue, and start a separate thread to continuously pull
    # messages from results queue to update final results.
    # Message in predictions queue should be a tuple of two elements:
    #    (evaluation step, predictions)
    # Message in results queue should be a tuple of two elements:
    #    (evaluation step, final results)
    self.async_eval_process = None
    self.async_eval_predictions_queue = None
    self.async_eval_results_queue = None
    self.async_eval_results_getter_thread = None

    # The MLPerf reference uses a starting lr of 1e-3 at bs=32.
    self.base_lr_batch_size = 32

  def skip_final_affine_layer(self):
    return True

  def postprocess(self, results):
    """Postprocess results returned from model."""
    try:
      import coco_metric  # pylint: disable=g-import-not-at-top
    except ImportError:
      raise ImportError('To use the COCO dataset, you must clone the '
                        'repo https://github.com/tensorflow/models and add '
                        'tensorflow/models and tensorflow/models/research to '
                        'the PYTHONPATH, and compile the protobufs by '
                        'following https://github.com/tensorflow/models/blob/'
                        'master/research/object_detection/g3doc/installation.md'
                        '#protobuf-compilation ; To evaluate using COCO'
                        'metric, download and install Python COCO API from'
                        'https://github.com/cocodataset/cocoapi')

    pred_boxes = results[ssd_constants.PRED_BOXES]
    pred_scores = results[ssd_constants.PRED_SCORES]
    # TODO(haoyuzhang): maybe use these values for visualization.
    # gt_boxes = results['gt_boxes']
    # gt_classes = results['gt_classes']
    source_id = results[ssd_constants.SOURCE_ID]
    raw_shape = results[ssd_constants.RAW_SHAPE]

    # COCO evaluation requires processing COCO_NUM_VAL_IMAGES exactly once. Due
    # to rounding errors (i.e., COCO_NUM_VAL_IMAGES % batch_size != 0), setting
    # `num_eval_epochs` to 1 is not enough and will often miss some images. We
    # expect user to set `num_eval_epochs` to >1, which will leave some unused
    # images from previous steps in `predictions`. Here we check if we are doing
    # eval at a new global step.
    if results['global_step'] > self.eval_global_step:
      self.eval_global_step = results['global_step']
      self.predictions.clear()

    for i, sid in enumerate(source_id):
      self.predictions[int(sid)] = {
          ssd_constants.PRED_BOXES: pred_boxes[i],
          ssd_constants.PRED_SCORES: pred_scores[i],
          ssd_constants.SOURCE_ID: source_id[i],
          ssd_constants.RAW_SHAPE: raw_shape[i]
      }

    # COCO metric calculates mAP only after a full epoch of evaluation. Return
    # dummy results for top_N_accuracy to be compatible with benchmar_cnn.py.
    if len(self.predictions) >= ssd_constants.COCO_NUM_VAL_IMAGES:
      print('Got results for all {:d} eval examples. Calculate mAP...'.format(
          ssd_constants.COCO_NUM_VAL_IMAGES))

      annotation_file = os.path.join(self.data_dir,
                                     ssd_constants.ANNOTATION_FILE)
      # Size of predictions before decoding about 15--30GB, while size after
      # decoding is 100--200MB. When using async eval mode, decoding takes
      # 20--30 seconds of main thread time but is necessary to avoid OOM during
      # inter-process communication.
      decoded_preds = coco_metric.decode_predictions(self.predictions.values())
      self.predictions.clear()

      eval_results = coco_metric.compute_map(decoded_preds, annotation_file)
      self.eval_coco_ap = eval_results['COCO/AP']
      ret = {'top_1_accuracy': self.eval_coco_ap, 'top_5_accuracy': 0.}
      return ret
    print('Got {:d} out of {:d} eval examples.'
           ' Waiting for the remaining to calculate mAP...'.format(
               len(self.predictions), ssd_constants.COCO_NUM_VAL_IMAGES))
    return {'top_1_accuracy': self.eval_coco_ap, 'top_5_accuracy': 0.}
