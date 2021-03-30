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

"""Send simulated image data to tensorflow_model_server loaded with ResNet50 or InceptionV3 model.

"""

from __future__ import print_function

import os
import random

import grpc
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf_v1
import time
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from util import preprocess_image, parse_example_proto

tf_v1.disable_eager_execution()

tf_v1.app.flags.DEFINE_string('server', 'localhost:8500', 'PredictionService host:port')
tf_v1.app.flags.DEFINE_integer('batch_size', 1, 'Batch size to use')
tf_v1.app.flags.DEFINE_string('data_dir', '', 'path to images in TF records format')
tf_v1.app.flags.DEFINE_string('model', 'resnet50v1_5', 'served model name')
FLAGS = tf_v1.app.flags.FLAGS
IMAGE_SIZE = 224


def sample_images(image_size):
    """Pull a random batch of images from FLAGS.data_dir containing TF record formatted ImageNet validation set
    Returns:
        ndarray of float32 with shape [FLAGS.batch_size, image_size, image_size, 3]
    """

    sample_file = random.choice(os.listdir(FLAGS.data_dir))
    dataset = tf.data.TFRecordDataset(os.path.join(FLAGS.data_dir, sample_file))
    dataset = dataset.map(lambda x: parse_example_proto(x)).shuffle(True).batch(FLAGS.batch_size)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    with tf.Session() as sess:
        images, labels = sess.run(next_element)
        images = np.array([sess.run(preprocess_image(x, FLAGS.model, image_size)) for x in images])

    return images


def main(_):
    channel = grpc.insecure_channel(FLAGS.server)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    i = 0
    num_iteration = 40
    warm_up_iteration = 10
    total_time = 0
    for _ in range(num_iteration):
        i += 1
        if FLAGS.data_dir:
            image_np = sample_images(IMAGE_SIZE)
        else:
            image_np = np.random.rand(FLAGS.batch_size, IMAGE_SIZE, IMAGE_SIZE, 3).astype(np.float32)
            image_np *= 256.0

        request = predict_pb2.PredictRequest()
        request.model_spec.name = FLAGS.model
        request.model_spec.signature_name = 'serving_default'
        request.inputs['input_tensor'].CopyFrom(
            tf.make_tensor_proto(image_np, shape=[FLAGS.batch_size, IMAGE_SIZE, IMAGE_SIZE, 3]))
        start_time = time.time()
        stub.Predict(request, 500.0)  # 500 sec timeout
        time_consume = time.time() - start_time
        print('Iteration %d: %.3f sec' % (i, time_consume))
        if i > warm_up_iteration:
            total_time += time_consume

    time_average = total_time / (num_iteration - warm_up_iteration)
    print("Total: {} Average: {:.3f}".format(total_time, time_average))

    print('Batch size = %d' % FLAGS.batch_size)
    if (FLAGS.batch_size == 1):
        print('Latency: %.3f ms' % (time_average * 1000))

    print('Throughput: %.3f images/sec' % (FLAGS.batch_size / time_average))


if __name__ == '__main__':
    tf_v1.app.run()
