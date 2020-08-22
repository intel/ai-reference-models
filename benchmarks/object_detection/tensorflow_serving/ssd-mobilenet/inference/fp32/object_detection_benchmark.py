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

#
#     USAGE     #
# python object_detection_benchmark.py -i <path-to-COCO-validation-images> -m <model> -p <protocol>


from __future__ import print_function
import argparse
import os
import time
import random
import requests
import numpy as np
from PIL import Image


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


def check_valid_model(value):
    """Verifies model name is supported"""
    if value not in ('rfcn', 'ssd-mobilenet'):
        raise argparse.ArgumentError("Model name {} does not match 'rfcn' or 'ssd-mobilenet'.".
                                     format(value))
    return value


def check_valid_protocol(value):
    """Verifies protocol is supported"""
    if value not in ('rest', 'grpc'):
        raise argparse.ArgumentError("Protocol name {} does not match 'rest' or 'grpc'.".
                                     format(value))
    return value


def get_random_image(image_dir):
    image_path = os.path.join(image_dir, random.choice(os.listdir(image_dir)))
    image = Image.open(image_path)
    (im_width, im_height) = image.size

    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def make_request(batch_size):
    if PROTOCOL == 'rest':
        np_images = np.repeat(np.expand_dims(get_random_image(IMAGES_PATH), 0).tolist(), batch_size, axis=0).tolist()
        return '{"instances" : %s}' % np_images
    elif PROTOCOL == 'grpc':
        import grpc
        import tensorflow as tf
        from tensorflow_serving.apis import predict_pb2
        from tensorflow_serving.apis import prediction_service_pb2_grpc
        np_images = np.repeat(np.expand_dims(get_random_image(IMAGES_PATH), 0), batch_size, axis=0)
        channel = grpc.insecure_channel(SERVER_URL)
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        request = predict_pb2.PredictRequest()
        request.model_spec.name = MODEL
        request.model_spec.signature_name = 'serving_default'
        request.inputs['inputs'].CopyFrom(tf.make_tensor_proto(np_images))
        return (stub, request)


def send_request(predict_request):
    if PROTOCOL == 'rest':
        requests.post(SERVER_URL, data=predict_request)
    elif PROTOCOL == 'grpc':
        predict_request[0].Predict(predict_request[1])


def benchmark(batch_size=1, num_iteration=20, warm_up_iteration=10):
    i = 0
    total_time = 0
    for _ in range(num_iteration):
        i += 1
        predict_request = make_request(batch_size)
        start_time = time.time()
        send_request(predict_request)
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
    ap.add_argument("-i", "--images_path", type=check_valid_folder, required=True,
                    help="Path to COCO validation directory")
    ap.add_argument("-m", "--model", type=check_valid_model, required=True,
                    help="Name of model (rfcn or ssd-mobilenet)")
    ap.add_argument("-p", "--protocol", type=check_valid_protocol, required=False, default="grpc",
                    help="Name of protocol (rest or grpc)")
    ap.add_argument("-b", "--batch_size", type=int, required=True,
                    help="Batch size")
    args = vars(ap.parse_args())

    IMAGES_PATH = args['images_path']
    MODEL = args['model']
    PROTOCOL = args['protocol']
    BATCH_SIZE = args['batch_size']
    if PROTOCOL == 'rest':
        SERVER_URL = 'http://localhost:8501/v1/models/{}:predict'.format(MODEL)
    elif PROTOCOL == 'grpc':
        SERVER_URL = 'localhost:8500'

    print('\n SERVER_URL: {} \n IMAGES_PATH: {}'.format(SERVER_URL, IMAGES_PATH))

    print('\nStarting {} model benchmarking for latency on {}:'.format(MODEL.upper(), PROTOCOL.upper()))
    print('batch_size=1, num_iteration=20, warm_up_iteration=10\n')
    benchmark(batch_size=BATCH_SIZE, num_iteration=20, warm_up_iteration=10)
