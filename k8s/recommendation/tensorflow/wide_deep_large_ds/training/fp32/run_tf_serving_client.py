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

tf.compat.v1.disable_eager_execution()


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
            raise argparse.ArgumentTypeError(
                "{} does not exist or is not a file.".
                format(value))
        check_for_link(value)
    return value


def generate_input_fn(filename, batch_size, num_epochs):
    """
    From train.py
    """
    CONTINUOUS_COLUMNS = ["I" + str(i) for i in range(1, 14)]  # 1-13 inclusive
    CATEGORICAL_COLUMNS = ["C" + str(i) for i in
                           range(1, 27)]  # 1-26 inclusive
    LABEL_COLUMN = ["clicked"]
    TRAIN_DATA_COLUMNS = LABEL_COLUMN + CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS

    def parse_csv(value):
        tf.compat.v1.logging.info('Parsing {}'.format(filename))
        cont_defaults = [[0.0] for i in range(1, 14)]
        cate_defaults = [[" "] for i in range(1, 27)]
        label_defaults = [[0]]
        column_headers = TRAIN_DATA_COLUMNS
        record_defaults = label_defaults + cont_defaults + cate_defaults
        columns = tf.io.decode_csv(value, record_defaults=record_defaults)
        all_columns = collections.OrderedDict(zip(column_headers, columns))
        labels = all_columns.pop(LABEL_COLUMN[0])
        features = all_columns
        return features, labels

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(filename)
    dataset = dataset.shuffle(buffer_size=20000)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.prefetch(batch_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(parse_csv, num_parallel_calls=28)
    dataset = dataset.prefetch(1)
    return dataset


def input_fn(data_file, num_epochs, shuffle, batch_size):
    """Generate an input function for the Estimator."""

    def _parse_function(proto):
        numeric_feature_names = ["numeric_1"]
        string_feature_names = ["string_1"]
        full_features_names = numeric_feature_names + string_feature_names + [
            "label"]
        feature_datatypes = [tf.io.FixedLenSequenceFeature([], tf.float32,
                                                           default_value=0.0,
                                                           allow_missing=True)] + [
                                tf.io.FixedLenSequenceFeature(
                                    [], tf.int64, default_value=0,
                                    allow_missing=True)] + [
                                tf.io.FixedLenSequenceFeature([], tf.int64,
                                                              default_value=0,
                                                              allow_missing=True)]
        f = collections.OrderedDict(
            zip(full_features_names, feature_datatypes))
        parsed_features = tf.io.parse_example(proto, f)
        parsed_feature_vals_num = [tf.reshape(
            parsed_features["numeric_1"], shape=[-1, 13])]
        parsed_feature_vals_str = [tf.reshape(
            parsed_features["string_1"], shape=[-1, 2]) for i in
            string_feature_names]
        parsed_feature_vals = parsed_feature_vals_num + parsed_feature_vals_str
        parsed_feature_vals_label = [tf.reshape(parsed_features[i], shape=[-1])
                                     for i in ["label"]]
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
    request.model_spec.name = 'wide-deep-large-ds-fp32-training'
    request.model_spec.signature_name = 'predict'

    i = 0
    total_time = 0
    res_dataset = generate_input_fn(DATA_FILE, batch_size=batch_size, num_epochs=1)
    #res_dataset = input_fn(DATA_FILE, 1, False, batch_size)

    CONTINUOUS_COLUMNS = ["I" + str(i) for i in range(1, 14)]  # 1-13 inclusive
    CATEGORICAL_COLUMNS = ["C" + str(i) for i in range(1, 27)]  # 1-26 inclusive
    LABEL_COLUMN = ["clicked"]
    TRAIN_DATA_COLUMNS = LABEL_COLUMN + CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
    FEATURE_COLUMNS = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS

    iterator = tf.compat.v1.data.make_one_shot_iterator(res_dataset)
    next_element = iterator.get_next()
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        try:
            for _ in range(num_iteration):
                correct_predictions = 0
                i += 1
                input_data = sess.run(next_element)

                for c in FEATURE_COLUMNS:
                    request.inputs[c].CopyFrom(
                        tf.make_tensor_proto(input_data[0][c], shape=[1 * batch_size]))

                # request.inputs['numeric'].CopyFrom(
                #     tf.make_tensor_proto(input_data[0],
                #                          shape=[1 * batch_size, 13]))
                # request.inputs['categorical'].CopyFrom(
                #     tf.make_tensor_proto(input_data[1],
                #                          shape=[26 * batch_size, 2]))


                start_time = time.time()
                result = stub.Predict(request)
                # print("-" * 20 + " result " + "-" * 20)
                # print(result.outputs['probabilities'])
                # print("-" * 20 + " result " + "-" * 20)
                time_consume = time.time() - start_time
                if i > warm_up_iteration:
                    total_time += time_consume
                for j in range(batch_size):
                    if input_data[1][j] == np.argmax(
                            result.outputs['probabilities'].float_val[
                            j * 2:(j * 2) + 2]):
                        correct_predictions += 1
                print('Iteration %d: %.3f sec (accuracy: %.2f%%)' % (
                i, time_consume, 100.0 * correct_predictions / batch_size))

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
    ap.add_argument("-s", "--server", type=str, default='localhost:8500',
                    help="Server URL and port (default=localhost:8500).")
    ap.add_argument("-d", "--data_file", type=check_valid_file, required=True,
                    help="Path to csv data file")
    ap.add_argument("-b", "--batch_size", required=False, type=int, default=1,
                    help="Batch size to use (default=1).")
    ap.add_argument("-n", "--num_iteration", required=False, type=int,
                    default=40,
                    help="Number of times to repeat (default=40).")
    ap.add_argument("-w", "--warm_up_iteration", required=False, type=int,
                    default=10,
                    help="Number of initial iterations to ignore in "
                         "benchmarking (default=10).")

    args = vars(ap.parse_args())

    SERVER_URL = args['server']
    DATA_FILE = args['data_file']
    BATCH_SIZE = args['batch_size']
    NUM_ITERATION = args['num_iteration']
    WARM_UP_ITERATION = args['warm_up_iteration']

    print('\n SERVER_URL: {} \n DATA_FILE: {}'.format(SERVER_URL, DATA_FILE))

    print('\nStarting Wide and Deep model benchmarking for Latency with '
          'batch_size={}, num_iteration=20, warm_up_iteration=10'.format(
        BATCH_SIZE))
    benchmark(batch_size=BATCH_SIZE, num_iteration=NUM_ITERATION,
              warm_up_iteration=WARM_UP_ITERATION)
