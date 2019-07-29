#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: EPL-2.0
#
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Infers detections on a TFRecord of TFExamples given an inference graph.

Example usage:
    ./infer_detections \
    --input_tfrecord_paths=/path/to/input/tfrecord1,/path/to/input/tfrecord2 \
    --output_tfrecord_path_prefix=/path/to/output/detections.tfrecord \
    --inference_graph=/path/to/frozen_weights_inference_graph.pb

The output is a TFRecord of TFExamples. Each TFExample from the input is first
augmented with detections from the inference graph and then copied to the
output.

The input and output nodes of the inference graph are expected to have the same
types, shapes, and semantics, as the input and output nodes of graphs produced
by export_inference_graph.py, when run with --input_type=image_tensor.

The script can also discard the image pixels in the output. This greatly
reduces the output size and can potentially accelerate reading data in
subsequent processing steps that don't require the images (e.g. computing
metrics).
"""

import itertools
import tensorflow as tf
from object_detection.inference import detection_inference
import numpy as np
import time
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--input_tfrecord_paths', type=str, default=None,
                    help='A comma separated list of paths to input TFRecords.')
parser.add_argument('--output_tfrecord_path', type=str, default=None,
                    help='Path to the output TFRecord.')
parser.add_argument('--inference_graph', type=str, default=None,
                    help='Path to the inference graph with embedded weights.')
parser.add_argument('--discard_image_pixels', type=bool, default=False,
                    help='Discards the images in the output TFExamples. This significantly reduces the output size '
                         'and is useful if the subsequent tools don\'t need access to the images '
                         '(e.g. when computing evaluation measures).')
parser.add_argument('--num_inter_threads', type=int, default=None,
                    help='Number of inter op threads')
parser.add_argument('--num_intra_threads', type=int, default=None,
                    help='Number of intra op threads')

args = parser.parse_args()


def main(_):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    required_flags = ['input_tfrecord_paths', 'output_tfrecord_path',
                      'inference_graph', 'num_inter_threads',
                      'num_intra_threads']
    for flag_name in required_flags:
        if not getattr(args, flag_name):
            raise ValueError('Flag --{} is required'.format(flag_name))

    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
            inter_op_parallelism_threads=args.num_inter_threads,
            intra_op_parallelism_threads=args.num_intra_threads)) as sess:
        input_tfrecord_paths = [
            v for v in args.input_tfrecord_paths.split(',') if v]
        tf.compat.v1.logging.info('Reading input from %d files', len(input_tfrecord_paths))
        serialized_example_tensor, image_tensor = detection_inference.build_input(
            input_tfrecord_paths)
        tf.compat.v1.logging.info('Reading graph and building model...')
        (detected_boxes_tensor, detected_scores_tensor,
         detected_labels_tensor) = detection_inference.build_inference_graph(
            image_tensor, args.inference_graph)

        tf.compat.v1.logging.info('Running inference and writing output to {}'.format(
            args.output_tfrecord_path))
        sess.run(tf.compat.v1.local_variables_initializer())
        tf.compat.v1.train.start_queue_runners()

        latency = []
        with tf.io.TFRecordWriter(
                args.output_tfrecord_path) as tf_record_writer:
            try:
                for counter in itertools.count():
                    tf.compat.v1.logging.log_every_n(
                        tf.compat.v1.logging.INFO,
                        'Processed %d images... moving average latency %d ms',
                        200, counter + 1, np.mean(latency[-200:]))
                    start = time.time()
                    tf_example = detection_inference.\
                        infer_detections_and_add_to_example(
                            serialized_example_tensor, detected_boxes_tensor,
                            detected_scores_tensor, detected_labels_tensor,
                            args.discard_image_pixels)
                    duration = time.time() - start
                    latency.append(duration * 1000)
                    tf_record_writer.write(tf_example.SerializeToString())
            except tf.errors.OutOfRangeError:
                tf.compat.v1.logging.info('Finished processing records')
        latency = np.array(latency)
        print("Latency: min = {:.1f}, max = {:.1f}, mean= {:.1f}, median "
              "= {:.1f}".format(latency.min(), latency.max(), latency.mean(),
                                np.median(latency)))


if __name__ == '__main__':
    tf.compat.v1.app.run()
