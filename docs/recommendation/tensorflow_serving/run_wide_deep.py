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

# Usage: python wide_deep_benchmark.py -d <path-to-tfrecords-data-file>


from __future__ import print_function

import os
import time
import collections
import argparse
import grpc
import numpy as np
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


def check_for_link(value):
    """
    Throws an error if the specified path is a link. os.islink returns
    True for sym links.  For files, we also look at the number of links in
    os.stat() to determine if it's a hard link.
    """
    if os.path.islink(value) or \
            (os.path.isfile(value) and os.stat(value).st_nlink > 1):
        raise argparse.ArgumentTypeError("{} cannot be a link.".format(value))

def check_valid_file(value):
    """verifies filename exists and isn't a link"""
    if value is not None:
        if not os.path.isfile(value):
            raise argparse.ArgumentTypeError("{} does not exist or is not a file.".
                                    format(value))
        check_for_link(value)
    return value

def input_fn(data_file, num_epochs, shuffle, batch_size):
    """Generate an input function for the Estimator."""
    def _parse_function(proto):
        numeric_feature_names = ["numeric_1"]
        string_feature_names = ["string_1"]
        full_features_names = numeric_feature_names + string_feature_names + ["label"]
        feature_datatypes = [tf.FixedLenSequenceFeature([], tf.float32, default_value=0.0, allow_missing=True)]+[tf.FixedLenSequenceFeature(
            [], tf.int64, default_value=0, allow_missing=True)]+[tf.FixedLenSequenceFeature([], tf.int64, default_value=0, allow_missing=True)]
        f = collections.OrderedDict(zip(full_features_names, feature_datatypes))
        parsed_features = tf.parse_example(proto, f)
        parsed_feature_vals_num = [tf.reshape(
            parsed_features["numeric_1"], shape=[-1, 13])]
        parsed_feature_vals_str = [tf.reshape(
            parsed_features["string_1"], shape=[-1, 2]) for i in string_feature_names]
        parsed_feature_vals = parsed_feature_vals_num + parsed_feature_vals_str
        parsed_feature_vals_label = [tf.reshape(parsed_features[i], shape=[-1]) for i in ["label"]]
        parsed_feature_vals = parsed_feature_vals + parsed_feature_vals_label
        return parsed_feature_vals

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TFRecordDataset([data_file])
    if shuffle:
        dataset = dataset.shuffle(buffer_size=20000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(_parse_function, num_parallel_calls=16)
    return dataset

def benchmark(batch_size=1, num_iteration=20, warm_up_iteration=10):
    channel = grpc.insecure_channel(SERVER_URL)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'wide_deep'
    request.model_spec.signature_name = 'serving_default'

    i = 0
    total_time = 0
    res_dataset = input_fn(DATA_FILE, 1, False, batch_size)
    iterator = res_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try: 
            for _ in range(num_iteration):
                correct_predictions = 0
                i += 1
                input_data = sess.run(next_element)
                request.inputs['numeric'].CopyFrom(tf.contrib.util.make_tensor_proto(input_data[0], shape=[1 * batch_size, 13]))
                request.inputs['categorical'].CopyFrom(tf.contrib.util.make_tensor_proto(input_data[1], shape=[26 * batch_size, 2]))

                start_time = time.time()
                result = stub.Predict(request)
                time_consume = time.time() - start_time
                if i > warm_up_iteration:
                    total_time += time_consume
                for j in range(batch_size):
                    if input_data[2][j] == np.argmax(result.outputs['output'].float_val[j*2:(j*2)+2]):
                        correct_predictions += 1
                print('Iteration %d: %.3f sec (accuracy: %.2f%%)' % (i, time_consume, 100.0*correct_predictions/batch_size))

        except tf.errors.OutOfRangeError:
            pass 

    time_average = total_time / (num_iteration - warm_up_iteration)
    print('Average time: %.3f sec' % (time_average))
    print('Batch size = %d' % batch_size)
    if batch_size == 1:
        print('Latency: %.3f ms' % (time_average * 1000))
    print('Throughput: %.3f examples/sec' % (batch_size / time_average))

    
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data_file", type=check_valid_file, required=True,
                    help="Path to TFRecords data file")
    ap.add_argument("-b", "--batch_size", required=False, type=int, default=1,
                    help="Batch size to use")
    ap.add_argument("-n", "--num_iteration", required=False, type=int, default=40,
                    help="Number of times to repeat")
    ap.add_argument("-w", "--warm_up_iteration", required=False, type=int, default=10,
                    help="Number of initial iterations to ignore in benchmarking")

    args = vars(ap.parse_args())

    SERVER_URL = 'localhost:8500'
    DATA_FILE = args['data_file']
    BATCH_SIZE = args['batch_size']
    NUM_ITERATION = args['num_iteration']
    WARM_UP_ITERATION = args['warm_up_iteration']

    print('\n SERVER_URL: {} \n DATA_FILE: {}'.format(SERVER_URL, DATA_FILE))
    
    print('\nStarting Wide and Deep model benchmarking for Latency with batch_size={}, num_iteration=20, warm_up_iteration=10'.format(BATCH_SIZE))
    benchmark(batch_size=BATCH_SIZE, num_iteration=NUM_ITERATION, warm_up_iteration=WARM_UP_ITERATION)
    
