#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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

import time
from argparse import ArgumentParser

import tensorflow as tf
import tensorflow_gnn as tfgnn
from tensorflow.python.platform import tf_logging

from utils import generate_dataset


class TimeHistory(tf.keras.callbacks.Callback):
    def __init__(self, batch_size, warmup_steps):
        self.times = []
        self.throughput = []
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps

    def on_predict_batch_begin(self, batch, logs={}):
        tf_logging.warn('\n---> Start iteration {0}'.format(str(batch)))
        self.epoch_time_start = time.time()

    def on_predict_batch_end(self, batch, logs={}):
        tf_logging.warn('\n---> Stop iteration {0}'.format(str(batch)))
        if ( batch >= self.warmup_steps):
            total_time = time.time() - self.epoch_time_start
            self.times.append(total_time)
            self.throughput.append(self.batch_size / total_time)

    def on_test_batch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_test_batch_end(self, batch, logs={}):
        total_time = time.time() - self.epoch_time_start
        self.times.append(total_time)
        self.throughput.append(self.batch_size / total_time)


class eval_rgat:
    """Evaluate RGAT model"""

    def __init__(self):

        arg_parser = ArgumentParser(description='Parse args')

        arg_parser.add_argument('-b', "--batch-size",
                                help='Specify the batch size. If this '
                                     'parameter is not specified, then '
                                     'it will run with batch size of 1000 ',
                                dest='batch_size', type=int, default=1000)

        arg_parser.add_argument("-p", "--precision",
                                help="Specify the model precision to use: fp32, bfloat16 or fp16",
                                required=True, choices=["fp32", "bfloat16", "fp16"],
                                dest="precision")

        arg_parser.add_argument('-e', "--num-inter-threads",
                                help='The number of inter-thread.',
                                dest='num_inter_threads', type=int, default=0)

        arg_parser.add_argument('-a', "--num-intra-threads",
                                help='The number of intra-thread.',
                                dest='num_intra_threads', type=int, default=0)

        arg_parser.add_argument('-g', "--graph-schema-path",
                                help='Specify the path to the graph schema',
                                dest='graph_schema_path', required=True)

        arg_parser.add_argument('-m', "--pretrained-model",
                                help='Specify the path to the pretrained model',
                                dest='pretrained_model', required=True)

        arg_parser.add_argument('-d', "--data-location",
                                help='Specify the location of the data. ',
                                dest="data_location", required=True)

        arg_parser.add_argument('-r', "--accuracy-only",
                                help='For accuracy measurement only.',
                                dest='accuracy_only', action='store_true')

        arg_parser.add_argument("--warmup-steps", type=int, default=10,
                                help="number of warmup steps")

        arg_parser.add_argument('-s', "--steps", type=int, default=200,
                                help="number of steps")

        # parse the arguments
        self.args = arg_parser.parse_args()
        # validate the arguements
        self.validate_args()

    def run(self):
        print("Run inference with " + str(self.args.precision) + " precision")

        if self.args.precision == "bfloat16":
            print("Enabling auto-mixed precision for bfloat16")
            tf.config.optimizer.set_experimental_options({'auto_mixed_precision_onednn_bfloat16': True})
            print(tf.config.optimizer.get_experimental_options())
        elif self.args.precision == "fp16":
            print("Enabling auto-mixed precision for fp16")
            tf.config.optimizer.set_experimental_options({'auto_mixed_precision': True})
            print(tf.config.optimizer.get_experimental_options())

        graph_schema = tfgnn.read_schema(self.args.graph_schema_path)
        print("*** Graph Schema ***")
        print(graph_schema)

        graph_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)

        datasets = generate_dataset(graph_spec, self.args.data_location, self.args.batch_size)

        model = tf.keras.models.load_model(self.args.pretrained_model)

        if (not self.args.accuracy_only):
            print("Benchmark")

            time_callback = TimeHistory(self.args.batch_size, self.args.warmup_steps)
            model.predict(datasets["test"], batch_size=self.args.batch_size, callbacks=[time_callback], steps=self.args.steps)

            avg_time = sum(time_callback.times) / len(time_callback.times)
            avg_throughput = sum(time_callback.throughput) / len(time_callback.throughput)

            print('Batch size = %d' % self.args.batch_size)
            if (self.args.batch_size == 1):
                print('Latency: %.3f ms' % (avg_time * 1000))
            # print throughput for both batch size 1 and batch_size
            print("Avg Throughput: " + str(avg_throughput) + " examples/sec")
        else:  # accuracy check
            print("Accuracy")

            time_callback = TimeHistory(self.args.batch_size, self.args.warmup_steps)
            test_scores = model.evaluate(datasets["test"], batch_size=self.args.batch_size,
                                         callbacks=[time_callback])
            avg_time = sum(time_callback.times) / len(time_callback.times)
            print("Avg evaluation time: %0.4f ms" % avg_time)
            print("Test loss: {}".format(test_scores[0]))
            print("Test accuracy: {}".format(test_scores[1]))

    def validate_args(self):
        """validate the arguments"""

        if not self.args.data_location:
            if self.args.accuracy_only:
                raise ValueError("You must use real data for accuracy measurement.")


if __name__ == "__main__":
    evaluate_rgat = eval_rgat()
    evaluate_rgat.run()
