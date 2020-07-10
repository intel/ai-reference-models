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
import argparse
import grpc
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from utils import tokenizer
from utils.tokenizer import Subtokenizer


def check_for_link(value):
    """
    Throws an error if the specified path is a link. os.islink returns
    True for sym links.  For files, we also look at the number of links in
    os.stat() to determine if it's a hard link.
    """
    if os.path.islink(value) or \
            (os.path.isfile(value) and os.stat(value).st_nlink > 1):
        raise argparse.ArgumentTypeError("{} cannot be a link.".format(value))


def check_valid_file_or_folder(value):
    """verifies filename exists and isn't a link"""
    if value is not None:
        if not os.path.isfile(value) and not os.path.isdir(value):
            raise argparse.ArgumentTypeError("{} does not exist or is not a file/folder.".
                                             format(value))
        check_for_link(value)
    return value


def input_generator_ts(file_path, vocab_file):
    """Read and sort lines based on token count from the file
    sorted by decreasing length based on token sorting.

    Args:
        file_path: String path of file to read
        vocab_file: String path of vocab file
    Returns:
        Sorted list of inputs, and dictionary mapping original index->sorted index
        of each element.
    """
    with tf.io.gfile.GFile(file_path) as f:
        records = f.read().split("\n")
        inputs = [record.strip() for record in records]
        if not inputs[-1]:
            inputs.pop()

    subtokenizer = Subtokenizer(vocab_file)

    batch = []
    token_lens = []
    for i, line in enumerate(inputs):
        enc = subtokenizer.encode(line, add_eos=True)
        token_lens.append((i, len(enc)))

    sorted_by_token_input_lens = sorted(token_lens, key=lambda x: x[1], reverse=True)
    sorted_inputs = [None] * len(sorted_by_token_input_lens)
    sorted_keys = [0] * len(sorted_by_token_input_lens)

    for i, (index, _) in enumerate(sorted_by_token_input_lens):
        sorted_inputs[i] = inputs[index]
        sorted_keys[index] = i
        enc = subtokenizer.encode(sorted_inputs[i], add_eos=True)
        batch.append(enc)

    return batch, sorted_keys


def _trim_and_decode(ids, vocab_file):
    """Trim EOS and PAD tokens from ids, and decode to return a string."""
    subtokenizer = Subtokenizer(vocab_file)
    try:
        index = list(ids).index(tokenizer.EOS_ID)
        return subtokenizer.decode(ids[:index])
    except ValueError:  # No EOS found in sequence
        return subtokenizer.decode(ids)


def benchmark(batch_size=1, num_iteration=20, warm_up_iteration=10):
    channel = grpc.insecure_channel(SERVER_URL)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'transformer_lt_official'
    request.model_spec.signature_name = 'serving_default'

    batches, sorted_keys = input_generator_ts(DATA_FILE, VOCAB_FILE)

    translations = []
    batch = []
    inference_time = 0.0
    sentences_to_translate = min(batch_size * num_iteration, len(batches))
    sentences_after_warmup = 0

    for i, line in enumerate(batches[0:sentences_to_translate]):
        batch.append(line)
        if (i + 1) % batch_size == 0 or i == sentences_to_translate - 1:
            batch_num = (i // batch_size) + 1
            request.inputs['input'].CopyFrom(
                tf.make_tensor_proto(pd.DataFrame(batch).fillna(0).values.astype(np.int64)))
            start_time = time.time()
            result = stub.Predict(request)
            duration = time.time() - start_time
            shape = [int(dim.size) for dim in result.outputs['output'].tensor_shape.dim]
            translations += np.reshape(result.outputs['output'].int_val, shape).tolist()
            print('Iteration %d: %.3f sec' % (batch_num, duration))
            if batch_num > warm_up_iteration:
                inference_time += duration
                sentences_after_warmup += len(batch)
            batch = []

    average_time = inference_time / sentences_after_warmup
    print('Inferencing time: %s' % (inference_time))
    print('Batch size = %d' % batch_size)
    if batch_size == 1:
        print('Latency: %.3f ms' % (average_time * 1000))
    print('Throughput: %.3f sentences/sec' % (sentences_after_warmup / inference_time))

    if OUT_FILE:
        print('Decoding and saving translations to {}...'.format(OUT_FILE))
        decoded_translations = []
        for i, tr in enumerate(translations):
            decoded_translations.append(_trim_and_decode(tr, VOCAB_FILE))

        with tf.io.gfile.GFile(OUT_FILE, "w") as f:
            for i in sorted_keys:
                if i < len(decoded_translations):
                    f.write("%s\n" % decoded_translations[i])
        print('Done!')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--server", type=str, required=False, default='localhost:8500',
                    help="Server URL (default localhost:8500)")
    ap.add_argument("-d", "--data_file", type=check_valid_file_or_folder, required=True,
                    help="Path to English language input file")
    ap.add_argument("-v", "--vocab_file", type=check_valid_file_or_folder, required=True,
                    help="Path to vocabulary file")
    ap.add_argument("-o", "--out_file", type=str, required=False, default='',
                    help="Path to output file (optional)")
    ap.add_argument("-b", "--batch_size", required=False, type=int, default=1,
                    help="Batch size to use")
    ap.add_argument("-n", "--num_iteration", required=False, type=int, default=20,
                    help="Number of times to repeat")
    ap.add_argument("-w", "--warm_up_iteration", required=False, type=int, default=10,
                    help="Number of initial iterations to ignore in benchmarking")

    args = vars(ap.parse_args())

    SERVER_URL = args['server']
    DATA_FILE = args['data_file']
    VOCAB_FILE = args['vocab_file']
    OUT_FILE = args['out_file']
    BATCH_SIZE = args['batch_size']
    NUM_ITERATION = args['num_iteration']
    WARM_UP_ITERATION = args['warm_up_iteration']

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)

    print('\n SERVER_URL: {} \n DATA_FILE: {}'.format(SERVER_URL, DATA_FILE))

    print("\nStarting Transformer-LT (Official) model benchmarking for Latency with batch_size={}, "
          "num_iteration={}, warm_up_iteration={}".format(BATCH_SIZE, NUM_ITERATION, WARM_UP_ITERATION))
    benchmark(batch_size=BATCH_SIZE, num_iteration=NUM_ITERATION, warm_up_iteration=WARM_UP_ITERATION)
