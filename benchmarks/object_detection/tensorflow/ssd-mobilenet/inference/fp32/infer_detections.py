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


tf.flags.DEFINE_string('input_tfrecord_paths', None,
                       'A comma separated list of paths to input TFRecords.')
tf.flags.DEFINE_string('output_tfrecord_path', None,
                       'Path to the output TFRecord.')
tf.flags.DEFINE_string('inference_graph', None,
                       'Path to the inference graph with embedded weights.')
tf.flags.DEFINE_boolean('discard_image_pixels', False,
                        'Discards the images in the output TFExamples. This'
                        ' significantly reduces the output size and is useful'
                        ' if the subsequent tools don\'t need access to the'
                        ' images (e.g. when computing evaluation measures).')
tf.flags.DEFINE_integer('num_inter_threads', None,
                        'Number of inter op threads')
tf.flags.DEFINE_integer('num_intra_threads', None,
                        'Number of intra op threads')

FLAGS = tf.flags.FLAGS


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    required_flags = ['input_tfrecord_paths', 'output_tfrecord_path',
                      'inference_graph', 'num_inter_threads',
                      'num_intra_threads']
    for flag_name in required_flags:
        if not getattr(FLAGS, flag_name):
            raise ValueError('Flag --{} is required'.format(flag_name))

    with tf.Session(config=tf.ConfigProto(
            inter_op_parallelism_threads=FLAGS.num_inter_threads,
            intra_op_parallelism_threads=FLAGS.num_intra_threads)) as sess:
        input_tfrecord_paths = [
            v for v in FLAGS.input_tfrecord_paths.split(',') if v]
        tf.logging.info('Reading input from %d files', len(input_tfrecord_paths))
        serialized_example_tensor, image_tensor = detection_inference.build_input(
            input_tfrecord_paths)
        tf.logging.info('Reading graph and building model...')
        (detected_boxes_tensor, detected_scores_tensor,
         detected_labels_tensor) = detection_inference.build_inference_graph(
            image_tensor, FLAGS.inference_graph)

        tf.logging.info('Running inference and writing output to {}'.format(
            FLAGS.output_tfrecord_path))
        sess.run(tf.local_variables_initializer())
        tf.train.start_queue_runners()

        latency = []
        with tf.python_io.TFRecordWriter(
                FLAGS.output_tfrecord_path) as tf_record_writer:
            try:
                for counter in itertools.count():
                    tf.logging.log_every_n(
                        tf.logging.INFO,
                        'Processed %d images... moving average latency %d ms',
                        200, counter + 1, np.mean(latency[-200:]))
                    start = time.time()
                    tf_example = detection_inference.\
                        infer_detections_and_add_to_example(
                            serialized_example_tensor, detected_boxes_tensor,
                            detected_scores_tensor, detected_labels_tensor,
                            FLAGS.discard_image_pixels)
                    duration = time.time() - start
                    latency.append(duration * 1000)
                    tf_record_writer.write(tf_example.SerializeToString())
            except tf.errors.OutOfRangeError:
                tf.logging.info('Finished processing records')
        latency = np.array(latency)
        print("Latency: min = {:.1f}, max = {:.1f}, mean= {:.1f}, median "
              "= {:.1f}".format(latency.min(), latency.max(), latency.mean(),
                                np.median(latency)))


if __name__ == '__main__':
    tf.app.run()
