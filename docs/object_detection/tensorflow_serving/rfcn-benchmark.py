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

####### USAGE #########
# python rfcn-benchmark.py -i <path-to-COCO-validation-images>


from __future__ import print_function
import argparse
import os
import time
import random
import requests
import numpy as np
from PIL import Image
import tensorflow as tf
from object_detection.utils.visualization_utils import visualize_boxes_and_labels_on_image_array


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
    """verifies filename exists and isn't a link"""
    if value is not None:
        if not os.path.isdir(value):
            raise argparse.ArgumentTypeError("{} does not exist or is not a directory.".
                                    format(value))
        check_for_link(value)
    return value

def get_random_image(image_dir):
    image_path = os.path.join(image_dir, random.choice(os.listdir(image_dir)))
    image = Image.open(image_path)
    (im_width, im_height) = image.size
    
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def benchmark(batch_size=1, num_iteration=20, warm_up_iteration=10):
    i = 0
    total_time = 0
    for _ in range(num_iteration):
        i += 1
        np_images = np.repeat(np.expand_dims(get_random_image(IMAGES_PATH), 0).tolist(), batch_size, axis=0).tolist()
        predict_request = '{"instances" : %s}' % np_images
        start_time = time.time()
        requests.post(SERVER_URL, data=predict_request)
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
    args = vars(ap.parse_args())
    
    SERVER_URL = 'http://localhost:8501/v1/models/rfcn:predict'
    IMAGES_PATH = args['images_path']

    print('\n SERVER_URL: {} \n IMAGES_PATH: {}'.format(SERVER_URL, IMAGES_PATH))
    
    print('\nStarting R-FCN model benchmarking for Latency with batch_size=1, num_iteration=20, warm_up_iteration=10')
    benchmark(batch_size=1, num_iteration=20, warm_up_iteration=10)
    
    print('\nStarting R-FCN model benchmarking for Throughput with batch_size=128, num_iteration=10, warm_up_iteration=2')
    benchmark(batch_size=128, num_iteration=10, warm_up_iteration=2)
