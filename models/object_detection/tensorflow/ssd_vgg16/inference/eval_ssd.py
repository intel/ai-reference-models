# Copyright 2018 Changan Wang

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import time
from argparse import ArgumentParser
import sys
from google.protobuf import text_format
import tensorflow as tf

from dataset import dataset_common
from preprocessing import ssd_preprocessing
import anchor_manipulator

SSD_VGG16_IMAGE_SIZE = 300
NUM_CLASSES = 81
NEGATIVE_RATIO = 1.0
SELECT_THRESHOLD = 0.1
MATCH_THRESHOLD = 0.5
NEG_THRESHOLD = 0.5
DATA_FORMAT = 'channels_last'
NUM_READERS = 10
NUM_PREPROCESSING_THREADS = 28


def input_fn(dataset_pattern='val-*', batch_size=1, data_location=None):
    out_shape = [SSD_VGG16_IMAGE_SIZE] * 2
    anchor_creator = anchor_manipulator.AnchorCreator(out_shape,
                                                      layers_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3),
                                                                     (1, 1)],
                                                      anchor_scales=[(0.1,), (0.2,), (0.375,), (0.55,), (0.725,),
                                                                     (0.9,)],
                                                      extra_anchor_scales=[(0.1414,), (0.2739,), (0.4541,), (0.6315,),
                                                                           (0.8078,), (0.9836,)],
                                                      anchor_ratios=[(1., 2., .5), (1., 2., 3., .5, 0.3333),
                                                                     (1., 2., 3., .5, 0.3333), (1., 2., 3., .5, 0.3333),
                                                                     (1., 2., .5), (1., 2., .5)],
                                                      layer_steps=[8, 16, 32, 64, 100, 300])
    all_anchors, all_num_anchors_depth, all_num_anchors_spatial = anchor_creator.get_all_anchors()

    num_anchors_per_layer = []
    for ind in range(len(all_anchors)):
        num_anchors_per_layer.append(all_num_anchors_depth[ind] * all_num_anchors_spatial[ind])

    anchor_encoder_decoder = anchor_manipulator.AnchorEncoder(allowed_borders=[1.0] * 6,
                                                              positive_threshold=MATCH_THRESHOLD,
                                                              ignore_threshold=NEG_THRESHOLD,
                                                              prior_scaling=[0.1, 0.1, 0.2, 0.2])

    image_preprocessing_fn = lambda image_, labels_, bboxes_: ssd_preprocessing.preprocess_image(image_, labels_,
                                                                                                 bboxes_, out_shape,
                                                                                                 is_training=False,
                                                                                                 data_format=DATA_FORMAT,
                                                                                                 output_rgb=False)
    anchor_encoder_fn = lambda glabels_, gbboxes_: anchor_encoder_decoder.encode_all_anchors(glabels_, gbboxes_,
                                                                                             all_anchors,
                                                                                             all_num_anchors_depth,
                                                                                             all_num_anchors_spatial)

    image, filename, shape, loc_targets, cls_targets, match_scores = \
        dataset_common.slim_get_batch(NUM_CLASSES,
                                      batch_size,
                                      'val',
                                      os.path.join(
                                          data_location,
                                          dataset_pattern),
                                      NUM_READERS,
                                      NUM_PREPROCESSING_THREADS,
                                      image_preprocessing_fn,
                                      anchor_encoder_fn,
                                      num_epochs=1,
                                      is_training=False)
	print(image, filename, shape, loc_targets, cls_targets, match_scores)
    return image, filename, shape


class EvaluateSSDModel():
    def __init__(self):

        arg_parser = ArgumentParser(description='Parse args')

        arg_parser.add_argument('-b', "--batch-size",
                                help="Specify the batch size. If this " \
                                     "parameter is not specified or is -1, the " \
                                     "largest ideal batch size for the model will " \
                                     "be used.",
                                dest="batch_size", type=int, default=1)

        arg_parser.add_argument('-e', "--num-inter-threads",
                                help='The number of inter-thread.',
                                dest='num_inter_threads', type=int, default=0)

        arg_parser.add_argument('-a', "--num-intra-threads",
                                help='The number of intra-thread.',
                                dest='num_intra_threads', type=int, default=0)

        arg_parser.add_argument('--data-num-inter-threads', dest='data_num_inter_threads',
                                help='number threads across operators',
                                type=int, default=21)

        arg_parser.add_argument('--data-num-intra-threads', dest='data_num_intra_threads',
                                help='number threads for data layer operator',
                                type=int, default=28)

        arg_parser.add_argument('--kmp-blocktime', dest='kmp_blocktime',
                                help='number of kmp blocktime',
                                type=int, default=1)

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

        arg_parser.add_argument("--warmup-steps", type=int, default=10,
                                help="number of warmup steps")

        arg_parser.add_argument("--steps", type=int, default=50,
                                help="number of steps")

        self.args = arg_parser.parse_args()

        os.environ["KMP_BLOCKTIME"] = str(self.args.kmp_blocktime)

    def eval(self):

        data_config = tf.ConfigProto()
        data_config.inter_op_parallelism_threads = self.args.data_num_inter_threads
        data_config.intra_op_parallelism_threads = self.args.data_num_intra_threads
        data_config.use_per_session_threads = 1

        infer_config = tf.ConfigProto()
        infer_config.inter_op_parallelism_threads = self.args.num_inter_threads  # self.args.num_inter_threads
        infer_config.intra_op_parallelism_threads = self.args.num_intra_threads  # self.args.num_intra_threads
        infer_config.use_per_session_threads = 1

        data_graph = tf.Graph()
        with data_graph.as_default():
            if self.args.data_location:  # real data
                image, filename, shape = \
				  print("**************************will run inout_fn*********************")
                    input_fn(dataset_pattern='val-*', batch_size=self.args.batch_size, data_location=self.args.data_location)
            else:  # dummy data
                input_shape = [self.args.batch_size, SSD_VGG16_IMAGE_SIZE, SSD_VGG16_IMAGE_SIZE, 3]
                image = tf.random.uniform(input_shape, -123.68, 151.06, dtype=tf.float32, name='synthetic_images')

        infer_graph = tf.Graph()
        model_file = self.args.input_graph
        with infer_graph.as_default():
            graph_def = tf.GraphDef()
            file_ext = os.path.splitext(model_file)[1]
            with open(model_file, "rb") as f:
                if file_ext == '.pbtxt':
                    text_format.Merge(f.read(), graph_def)
                else:
                    graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')

        # Define input and output Tensors for inference graph
        output_names = ["ExpandDims"]
        for i in range(1, 160):
            output_names.append("ExpandDims_" + str(i))

        input_operation = infer_graph.get_operation_by_name("input")
        output_operations = []
        for name in output_names:
            output_operations.append(infer_graph.get_operation_by_name(name).outputs[0])

        infer_sess = tf.Session(graph=infer_graph, config=infer_config)

        if not self.args.accuracy_only:  # benchmark
            step = 0
            total_steps = self.args.warmup_steps + self.args.steps

            total_images = 0
            total_duration = 0

            if not self.args.data_location: # inference with dummy data
                print("Inference with dummy data")
                data_sess = tf.Session(graph=data_graph, config=data_config)

                while step < total_steps:
                    step += 1
                    image_np = data_sess.run(image)
                    start_time = time.time()

                    infer_sess.run(output_operations, {input_operation.outputs[0]: image_np})
                    duration = time.time() - start_time

                    if step > self.args.warmup_steps:
                        total_duration += duration
                        total_images += self.args.batch_size
                    print('Iteration %d: %.6f sec' % (step, duration))
                    sys.stdout.flush()

            else: # benchmark with real data
                print("Inference with real data")
                with data_graph.as_default():
                    with tf.train.MonitoredTrainingSession(config=data_config) as data_sess:
                        while not data_sess.should_stop() and step < total_steps:
                            step += 1
                            start_time = time.time()
                            image_np, _, _ = data_sess.run([image, filename, shape])
                            infer_sess.run(output_operations, {input_operation.outputs[0]: image_np})
                            duration = time.time() - start_time

                            if step > self.args.warmup_steps:
                                total_duration += duration
                                total_images += self.args.batch_size
                            print('Iteration %d: %.6f sec' % (step, duration))
                            sys.stdout.flush()

            print('Batch size = %d' % self.args.batch_size)
            print('Throughput: %.3f images/sec' % (total_images / total_duration))
            if (self.args.batch_size == 1):
                latency = (total_duration / total_images) * 1000
                print('Latency: %.3f ms' % (latency))

        else: # accuracy only
            results = []
            filenames = []
            shapes = []
            total_processed_images = 0
            with data_graph.as_default():
                with tf.train.MonitoredTrainingSession(config=data_config) as data_sess:
                    while not data_sess.should_stop():
                        image_np, filename_np, shape_np = data_sess.run([image, filename, shape])
                        total_processed_images += self.args.batch_size
                        predict = infer_sess.run(output_operations, {input_operation.outputs[0]: image_np})
                        if (total_processed_images % 30 == 0):
                            print("Predicting results for {} images...".format(total_processed_images))
                            sys.stdout.flush()
                        results.append(predict)
                        filenames.append(filename_np[0])
                        shapes.append(shape_np[0])

            log_dir = os.path.join('./', 'logs')
            # if it doesn't exist, create.
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            for class_ind in range(1, NUM_CLASSES):
                with open(os.path.join(log_dir, 'results_{}.txt'.format(class_ind)), 'wt') as f:
                    for image_ind, pred in enumerate(results):
                        shape = shapes[image_ind]
                        filename = filenames[image_ind]
                        # parsing prediction results and calculate bbox
                        scores = pred[(class_ind * 2) - 2][0]
                        bboxes = pred[(class_ind * 2) - 1][0]
                        bboxes[:, 0] = (bboxes[:, 0] * shape[0]).astype(np.int32, copy=False) + 1
                        bboxes[:, 1] = (bboxes[:, 1] * shape[1]).astype(np.int32, copy=False) + 1
                        bboxes[:, 2] = (bboxes[:, 2] * shape[0]).astype(np.int32, copy=False) + 1
                        bboxes[:, 3] = (bboxes[:, 3] * shape[1]).astype(np.int32, copy=False) + 1

                        valid_mask = np.logical_and((bboxes[:, 2] - bboxes[:, 0] > 0),
                                                    (bboxes[:, 3] - bboxes[:, 1] > 0))

                        for det_ind in range(valid_mask.shape[0]):
                            if not valid_mask[det_ind]:
                                continue
                            f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                    format(filename.decode('utf8')[:-4], scores[det_ind],
                                           bboxes[det_ind, 1], bboxes[det_ind, 0],
                                           bboxes[det_ind, 3], bboxes[det_ind, 2]))

            coco_eval = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "validate_ssd_vgg16.py")
            cmd_prefix = "python " + coco_eval
            cmd_prefix += " --detections_path ./logs"
            cmd_prefix += " --annotations_file {}/instances_val2017.json".format(self.args.data_location)
            cmd = cmd_prefix
            os.system(cmd)

if __name__ == "__main__":
    obj = EvaluateSSDModel()
    obj.eval()
