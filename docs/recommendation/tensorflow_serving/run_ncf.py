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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import time
import math
import pickle
import shutil
import numpy as np

import grpc
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from official.recommendation import constants as rconst
from official.recommendation import data_preprocessing
from official.recommendation.ncf_main import get_hit_rate_and_ndcg

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
    """verifies folder exists and isn't a link"""
    if value is not None:
        if not os.path.isdir(value):
            raise argparse.ArgumentTypeError("{} does not exist or is not a folder.".
                                    format(value))
        check_for_link(value)
    return value

def generate_eval_dataset(data_dir, batch_size):
    ncf_dataset = data_preprocessing.construct_cache(
        dataset='ml-1m', data_dir=data_dir, 
        num_data_readers=1, match_mlperf=False)
    with open(ncf_dataset.cache_paths.eval_raw_file, 'rb') as f:
        eval_data = pickle.load(f)
    shutil.rmtree(ncf_dataset.cache_paths.cache_root)
    return eval_data[0]['user_id'], eval_data[0]['item_id'], eval_data[1], ncf_dataset.num_users

def run_inference(batch_size=1, num_iteration=20, warm_up_iteration=10):
    channel = grpc.insecure_channel(SERVER_URL)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'ncf'
    request.model_spec.signature_name = 'serving_default'
    users, items, labels, num_users = generate_eval_dataset(DATA_DIR, batch_size)

    predictions = []
    movies = []
    inference_time = 0.0
    compute_accuracy_metrics = False

    if num_iteration == 0:
        num_iteration = int(math.ceil(len(users)/float(batch_size)))
        compute_accuracy_metrics = True

    for i in range(num_iteration):
        begin_index = i * batch_size
        end_index = (i * batch_size) + batch_size
        request.inputs['user_id'].CopyFrom(tf.contrib.util.make_tensor_proto(
            users[begin_index:end_index].astype(np.int32)))
        request.inputs['item_id'].CopyFrom(tf.contrib.util.make_tensor_proto(
            items[begin_index:end_index].astype(np.uint16)))
        start_time = time.time()
        result = stub.Predict(request)
        duration = time.time() - start_time
        predictions.append(result.outputs['rating'].float_val)
        movies.append(result.outputs['item_id'].float_val)
        print('Iteration %d: %.3f sec' % (i, duration))
        if i > warm_up_iteration:
            inference_time += duration

    time_average = inference_time / (num_iteration - warm_up_iteration)
    print('Average time: %.3f sec' % (time_average))
    print('Batch size = %d' % batch_size)
    if batch_size == 1:
        print('Online inference performance: %.3f ms per sample' % (time_average * 1000))
    print('Batch inference performance: %.3f samples/sec' % (batch_size / time_average))

    if compute_accuracy_metrics:
        prediction_with_padding = np.concatenate(predictions, axis=0)
        predicted_scores_by_user = prediction_with_padding[
                               :num_users * (1 + rconst.NUM_EVAL_NEGATIVES)].reshape(num_users, -1)
        item_with_padding = np.concatenate(movies, axis=0)
        items_by_user = item_with_padding[:num_users * (1 + rconst.NUM_EVAL_NEGATIVES)].reshape(num_users, -1)

        hr, ndcg = get_hit_rate_and_ndcg(predicted_scores_by_user, items_by_user, match_mlperf=False)
        print('Accuracy: %.5f HR, %.5f NDCG' % (hr, ndcg))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data_dir", type=check_valid_folder, required=True,
                    help="Path to MovieLens ml_1m directory location")
    ap.add_argument("-b", "--batch_size", required=False, type=int, default=1,
                    help="Batch size to use")
    ap.add_argument("-n", "--num_iteration", required=False, type=int, default=0,
                    help="Number of times to repeat; default (0) will result in using the entire ML-1M evaluation dataset")
    ap.add_argument("-w", "--warm_up_iteration", required=False, type=int, default=10,
                    help="Number of initial iterations to ignore in benchmarking")

    args = vars(ap.parse_args())

    SERVER_URL = 'localhost:8500'
    DATA_DIR = args['data_dir']
    BATCH_SIZE = args['batch_size']
    NUM_ITERATION = args['num_iteration']
    WARM_UP_ITERATION = args['warm_up_iteration']

    print('\n SERVER_URL: {} \n DATA_DIR: {}'.format(SERVER_URL, DATA_DIR))
    
    print('\nStarting NCF model run for inference with batch_size={}, num_iteration={}, warm_up_iteration={}'.format(BATCH_SIZE, NUM_ITERATION, WARM_UP_ITERATION))
    run_inference(batch_size=BATCH_SIZE, num_iteration=NUM_ITERATION, warm_up_iteration=WARM_UP_ITERATION)

