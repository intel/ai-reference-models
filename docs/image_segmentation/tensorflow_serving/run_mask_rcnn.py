# Copyright (c) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


from __future__ import print_function

import os
import time
import random
import numpy as np
import skimage
import argparse
import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from config import Config
import model as modellib


class CocoConfig(Config):
    NAME = "coco"
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 81


class InferenceConfig(CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0


def check_for_link(value):
    """
    Throws an error if the specified path is a link. os.islink returns
    True for sym links.  For files, we also look at the number of links in
    os.stat() to determine if it's a hard link.
    """
    if os.path.islink(value) or \
            (os.path.isfile(value) and os.stat(value).st_nlink > 1):
        raise argparse.ArgumentTypeError("{} cannot be a link.".format(value))

def check_valid_folder(value):
    """Verifies filename exists and isn't a link"""
    if value is not None:
        if not os.path.isdir(value):
            raise argparse.ArgumentTypeError("{} does not exist or is not a directory.".
                                    format(value))
        check_for_link(value)
    return value

def benchmark(batch_size=1, num_iteration=20, warm_up_iteration=10):
    config = InferenceConfig()
    config.NUM_INTRA = 1
    config.NUM_INTER = 1

    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    channel = grpc.insecure_channel(SERVER_URL, options=[
        ('grpc.max_send_message_length', 50 * 1024 * 1024),
        ('grpc.max_receive_message_length', 50 * 1024 * 1024)
    ])
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'mask_rcnn'
    request.model_spec.signature_name = 'serving_default'

    i = 0
    total_time = 0

    for _ in range(num_iteration):
        i += 1
        file_names = next(os.walk(IMAGE_DIR))[2]
        image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
        molded_images, image_metas, windows = model.mold_inputs([image])
        request.inputs['input_image'].CopyFrom(tf.contrib.util.make_tensor_proto(molded_images.astype(np.float32)))
        request.inputs['input_image_meta'].CopyFrom(tf.contrib.util.make_tensor_proto(image_metas.astype(np.float32)))
        start_time = time.time()
        result = stub.Predict(request)
        time_consume = time.time() - start_time
        print('Iteration %d: %.3f sec' % (i, time_consume))
        if i > warm_up_iteration:
            total_time += time_consume

    time_average = total_time / (num_iteration - warm_up_iteration)
    print('Average time: %.3f sec' % (time_average))
    print('Batch size = %d' % batch_size)
    if batch_size == 1:
        print('Latency: %.3f ms' % (time_average * 1000))
    print('Throughput: %.3f images/sec' % (batch_size / time_average))

    
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_dir", type=check_valid_folder, required=True,
                    help="Path to Mask R-CNN model directory")
    ap.add_argument("-i", "--image_dir", type=check_valid_folder, required=True,
                    help="Path to COCO validation directory")
    ap.add_argument("-n", "--num_iteration", required=False, type=int, default=20,
                    help="Number of times to repeat")
    ap.add_argument("-w", "--warm_up_iteration", required=False, type=int, default=10,
                    help="Number of initial iterations to ignore in benchmarking")
    args = vars(ap.parse_args())

    MODEL_DIR = args['model_dir']
    IMAGE_DIR = args['image_dir']
    SERVER_URL = 'localhost:8500'
    NUM_ITERATION = args['num_iteration']
    WARM_UP_ITERATION = args['warm_up_iteration']

    print('\n SERVER_URL: {} \n IMAGES_PATH: {}'.format(SERVER_URL, IMAGE_DIR))
    
    print('\nRunning Mask R-CNN model with batch size 1:')
    print('batch_size=1, num_iteration={}, warm_up_iteration={}\n'.format(NUM_ITERATION, WARM_UP_ITERATION))
    benchmark(batch_size=1, num_iteration=NUM_ITERATION, warm_up_iteration=WARM_UP_ITERATION)

