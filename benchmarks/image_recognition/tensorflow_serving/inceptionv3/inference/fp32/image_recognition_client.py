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

"""Send JPEG image to tensorflow_model_server loaded with ResNet50 or InceptionV3 model.

"""

from __future__ import print_function

import grpc
import numpy as np
import requests
import sys
import tensorflow.compat.v1 as tf_v1
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from util import preprocess_image

# The image URL is the location of the image we should send to the server
IMAGE_URL = 'https://tensorflow.org/images/blogs/serving/cat.jpg'

tf_v1.app.flags.DEFINE_string('server', 'localhost:8500', 'PredictionService host:port')
tf_v1.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
tf_v1.app.flags.DEFINE_string('model', 'resnet50', 'Name of model (resnet50 or Inceptionv3).')
FLAGS = tf_v1.app.flags.FLAGS


def main(_):
    if FLAGS.model == 'resnet50':
        image_size = 224
    elif FLAGS.model == 'inceptionv3':
        image_size = 299
    else:
        print('Please specify model as either resnet50 or Inceptionv3.')
        sys.exit(-1)

    if FLAGS.image:
        with open(FLAGS.image, 'rb') as f:
            data = f.read()
    else:
        # Download the image URL if a path is not provided as input
        dl_request = requests.get(IMAGE_URL, stream=True)
        dl_request.raise_for_status()
        data = dl_request.content

    channel = grpc.insecure_channel(FLAGS.server)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = FLAGS.model
    request.model_spec.signature_name = 'serving_default'
    image_data = tf.reshape(preprocess_image(data, FLAGS.model, image_size), [1, image_size, image_size, 3])

    # Run the graph
    with tf_v1.Session() as sess:
        sess.run(tf_v1.global_variables_initializer())
        image_data = (sess.run(image_data))

    request.inputs['input'].CopyFrom(
        tf.make_tensor_proto(image_data, shape=[1, image_size, image_size, 3]))
    result = stub.Predict(request)
    print('Predicted class: ', str(np.argmax(result.outputs['predict'].float_val)))


if __name__ == '__main__':
    tf_v1.disable_eager_execution()
    tf_v1.app.run()
