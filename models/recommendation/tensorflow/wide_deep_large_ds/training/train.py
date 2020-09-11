#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Intel Corporation
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

#
'''This script used to create and train wide and deep model on Kaggle's Criteo Dataset'''

import time
import argparse
import tensorflow as tf
import math
import collections
import numpy as np
import os.path
from os import path
import sys
# Set to INFO for tracking training, default is WARN. ERROR for least messages
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
print("Using TensorFlow version %s" % (tf.__version__))
CONTINUOUS_COLUMNS = ["I"+str(i) for i in range(1, 14)]  # 1-13 inclusive
CATEGORICAL_COLUMNS = ["C"+str(i) for i in range(1, 27)]  # 1-26 inclusive
LABEL_COLUMN = ["clicked"]
TRAIN_DATA_COLUMNS = LABEL_COLUMN + CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
FEATURE_COLUMNS = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS


def generate_input_fn(filename, batch_size, num_epochs):
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


def build_feature_cols(train_file_path,test_file_path):
    # compute statistics(min,max,range) of train dataset
    print('****Computing statistics of train dataset*****')
    with open(train_file_path, 'r') as f, open(test_file_path, 'r') as f1:
        nums = [line.strip('\n').split(',') for line in f.readlines(
        )]+[line.strip('\n').split(',') for line in f1.readlines()]
        numpy_arr = np.array(nums)
        mins_list, max_list, range_list = [], [], []
        for i in range(len(TRAIN_DATA_COLUMNS)):
            if TRAIN_DATA_COLUMNS[i] in CONTINUOUS_COLUMNS:
                col_min = numpy_arr[:, i].astype(np.float32).min()
                col_max = numpy_arr[:, i].astype(np.float32).max()
                mins_list.append(col_min)
                max_list.append(col_max)
                range_list.append(col_max-col_min)

    def numeric_column_normalized(column_name, normalizer_fn):
        return tf.feature_column.numeric_column(column_name, normalizer_fn=normalizer_fn)

    def make_minmaxscaler(min, range):
        def minmaxscaler(col):
            return (col - min)/range
        return minmaxscaler
    deep_columns = []
    for i in range(len(CONTINUOUS_COLUMNS)):
        normalizer_fn = None
        col_min = mins_list[i]
        col_range = range_list[i]
        normalizer_fn = make_minmaxscaler(col_min, col_range)
        deep_columns.append(numeric_column_normalized(
            CONTINUOUS_COLUMNS[i], normalizer_fn))
    wide_columns = []
    for name in CATEGORICAL_COLUMNS:
        wide_columns.append(tf.feature_column.categorical_column_with_hash_bucket(
            name, hash_bucket_size=1000))
    for col in wide_columns:
        deep_columns.append(
            tf.feature_column.embedding_column(col, dimension=32))
    return wide_columns, deep_columns


def build_model(model_type, model_dir, wide_columns, deep_columns):
    runconfig = tf.estimator.RunConfig(
        save_checkpoints_steps=500
    )
    m = None
    # Linear Classifier
    if model_type == 'WIDE':
        m = tf.estimator.LinearClassifier(
            config=runconfig,
            model_dir=model_dir,
            feature_columns=wide_columns)

    # Deep Neural Net Classifier
    elif model_type == 'DEEP':
        m = tf.estimator.DNNClassifier(
            config=runconfig,
            model_dir=model_dir,
            feature_columns=deep_columns,
            hidden_units=[1024, 512, 256])

    # Combined Linear and Deep Classifier
    elif model_type == 'WIDE_AND_DEEP':
        m = tf.estimator.DNNLinearCombinedClassifier(
            config=runconfig,
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=[1024, 512, 256])

    return m


def build_estimator(model_type='WIDE_AND_DEEP', model_dir=None, train_file_path=None,test_file_path=None):
    if model_dir is None:
        model_dir = 'models/model_' + model_type + '_' + str(int(time.time()))
        print("Model directory = %s" % model_dir)

    wide_columns, deep_columns = build_feature_cols(train_file_path,test_file_path)
    m = build_model(model_type, model_dir, wide_columns, deep_columns)
    print('estimator built')
    return m


# All categorical columns are strings for this dataset
def column_to_dtype(column):
    if column in CATEGORICAL_COLUMNS:
        return tf.string
    else:
        return tf.float32


"""
  This function maps input columns (feature_placeholders) to 
  tensors that can be inputted into the graph 
  (similar in purpose to the output of our input functions)
  In this particular case, we need to accomodate the sparse fields (strings)
  so we have to do a slight modification to expand their dimensions, 
  just like in the input functions
"""


def serving_input_fn():
    feature_placeholders = {
        column: tf.compat.v1.placeholder(column_to_dtype(column), [None])
        for column in FEATURE_COLUMNS
    }
    # DNNCombinedLinearClassifier expects rank 2 Tensors,
    # but inputs should be rank 1, so that we can provide
    # scalars to the server
    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_placeholders.items()
    }
    return tf.estimator.export.ServingInputReceiver(
        features,  # input into graph
        feature_placeholders  # tensor input converted from request
    )


def train_and_eval():
    print("Begin training and evaluation")
    train_file = args.data_location+'/train.csv'
    test_file = args.data_location+'/eval.csv'
    if (not path.exists(train_file)) or (not path.exists(test_file)):
        print('------------------------------------------------------------------------------------------')
        print("train.csv or eval.csv does not exist in the given data_location. Please provide valid path")
        print('------------------------------------------------------------------------------------------')
        sys.exit() 
    no_of_training_examples = sum(1 for line in open(train_file))
    no_of_test_examples = sum(1 for line in open(test_file))
    batch_size = args.batch_size
    if args.steps == 0:
        no_of_epochs = 10
        train_steps = math.ceil(
        (float(no_of_epochs)*no_of_training_examples)/batch_size)
    else:
        no_of_epochs = math.ceil(
        (float(batch_size)*args.steps)/no_of_training_examples)
        train_steps = args.steps
    test_steps = math.ceil(float(no_of_test_examples)/batch_size)
    model_type = 'WIDE_AND_DEEP'
    model_dir = path.join(args.output_dir, 'model_' + model_type + '_' + str(int(time.time())))
    checkpoint_dir = args.checkpoint if args.checkpoint else model_dir
    print("Saving model checkpoints to " + checkpoint_dir)
    export_dir = args.output_dir
    m = build_estimator(model_type, checkpoint_dir, train_file, test_file)
    m.train(input_fn=lambda: generate_input_fn(
        train_file, batch_size, int(no_of_epochs)),steps=int(train_steps))
    print('fit done')
    results = m.evaluate(input_fn=lambda: generate_input_fn(
        test_file, batch_size, 1), steps=test_steps)
    print('evaluate done')

    export_folder = m.export_saved_model(
        export_dir,
        serving_input_fn
    )
    print('Model exported to ' + export_dir)
    print('Accuracy: %s' % results['accuracy'])


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_location',
        help='Full path of train data',
        required=True
    )
    parser.add_argument(
        '--steps',
        help='set the number of steps on train dataset.Default is will set to 1 epoch',
        type=int,
        default=0
    )
    parser.add_argument(
        '--batch_size',
        help='Batch size to train. Default is 512',
        type=int,
        default=512
    )
    parser.add_argument(
        '--output_dir',
        help='Full path to output directory for logs and saved model',
        required=True
    )
    parser.add_argument(
        '--checkpoint',
        help='Full path to the input/output directory for checkpoints',
        required=False
    )
    return parser


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    main_start = time.time()
    train_and_eval()
    main_end = time.time()
    print("Total time:", main_end-main_start)
