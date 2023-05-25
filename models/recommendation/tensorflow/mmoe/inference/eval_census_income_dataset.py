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

#

import time
from argparse import ArgumentParser

import tensorflow as tf
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.core.protobuf import rewriter_config_pb2

from sklearn.metrics import roc_auc_score
from utils import preprocess_data, fetch_batch

CENSUS_INCOME_VALIDATION_DATA_RECORDS = 99762


class eval_classifier_optimized_graph:
    """Evaluate image classifier with optimized TensorFlow graph"""

    def __init__(self):

        arg_parser = ArgumentParser(description='Parse args')

        arg_parser.add_argument('-b', "--batch-size",
                                help='Specify the batch size. If this '
                                     'parameter is not specified, then '
                                     'it will run with batch size of 256 ',
                                dest='batch_size', type=int, default=256)

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

        arg_parser.add_argument('-g', "--input-graph",
                                help='Specify the input graph for the transform tool',
                                dest='input_graph', required=True)

        arg_parser.add_argument('-d', "--data-location",
                                help='Specify the location of the data. '
                                     'If this parameter is not specified, '
                                     'the benchmark will use random/dummy data.',
                                dest="data_location", default=None)

        arg_parser.add_argument('-r', "--accuracy-only",
                                help='For accuracy measurement only.',
                                dest='accuracy_only', action='store_true')

        arg_parser.add_argument("--warmup-steps", type=int, default=20,
                                help="number of warmup steps")
                                
        arg_parser.add_argument("--steps", type=int, default=200,
                                help="number of steps")

        # parse the arguments
        self.args = arg_parser.parse_args()
        # validate the arguements
        self.validate_args()

    def run(self):
        print("Running inference with " + str(self.args.precision) + " precision and batch size of " + str(self.args.batch_size))

        infer_config = tf.compat.v1.ConfigProto()
        infer_config.intra_op_parallelism_threads = self.args.num_intra_threads
        infer_config.inter_op_parallelism_threads = self.args.num_inter_threads
        infer_config.use_per_session_threads = 1

        if self.args.precision == "bfloat16":
            print("Enabling auto-mixed precision for bfloat16")
            infer_config.graph_options.rewrite_options.auto_mixed_precision_onednn_bfloat16 = rewriter_config_pb2.RewriterConfig.ON
        if self.args.precision == "fp16":
            print("Enabling auto-mixed precision for fp16")
            infer_config.graph_options.rewrite_options.auto_mixed_precision = rewriter_config_pb2.RewriterConfig.ON

        # Load the frozen model
        sm = saved_model_pb2.SavedModel()
        with tf.io.gfile.GFile(self.args.input_graph, "rb") as f:
            sm.ParseFromString(f.read())
        g_def = sm.meta_graphs[0].graph_def
        with tf.Graph().as_default() as infer_graph:
            tf.import_graph_def(g_def, name='')

        test_data, test_labels = None, None
        if (self.args.data_location):
            print("Inference with real data.")
            test_data, test_labels = preprocess_data(self.args.data_location)

        output_tensor = [infer_graph.get_tensor_by_name('Identity:0'), infer_graph.get_tensor_by_name('Identity_1:0')]

        infer_sess = tf.compat.v1.Session(graph=infer_graph, config=infer_config)

        num_processed_records = 0
        num_remaining_records = CENSUS_INCOME_VALIDATION_DATA_RECORDS

        if (not self.args.accuracy_only):
            print("Benchmark")
            iteration = 0
            warm_up_iteration = self.args.warmup_steps
            total_run = self.args.steps
            total_time = 0

            while iteration < total_run:
                iteration += 1

                # Reads and preprocess data
                # data_load_start = time.time()
                input_dict = fetch_batch(infer_graph, batch_size=self.args.batch_size, data=test_data,
                                         labels=test_labels)
                # data_load_time = time.time() - data_load_start

                start_time = time.time()
                predictions = infer_sess.run(output_tensor, feed_dict=input_dict)
                time_consume = time.time() - start_time

                # only add data loading time for real data, not for dummy data
                # if self.args.data_location:
                #   time_consume += data_load_time

                print('Iteration %d: %.6f sec' % (iteration, time_consume))
                if iteration > warm_up_iteration:
                    total_time += time_consume

            time_average = total_time / (iteration - warm_up_iteration)
            print('Average time: %.6f sec' % (time_average))

            print('Batch size = %d' % self.args.batch_size)
            if (self.args.batch_size == 1):
                print('Latency: %.3f ms' % (time_average * 1000))
            # print throughput for both batch size 1 and batch_size
            print('Throughput: %.3f examples/sec' % (self.args.batch_size / time_average))
        else:  # accuracy check
            total_income_auc, total_marital_stat_auc = 0.0, 0.0
            elapsed_time = 0.0
            iterations = 0
            while num_remaining_records >= self.args.batch_size:
                # Prepare a batch of data
                input_dict = fetch_batch(infer_graph, batch_size=self.args.batch_size, data=test_data,
                                         labels=test_labels, i=iterations)

                num_processed_records += self.args.batch_size
                num_remaining_records -= self.args.batch_size

                start_time = time.time()
                predictions = infer_sess.run(output_tensor, feed_dict=input_dict)
                elapsed_time += time.time() - start_time

                label_income = test_labels['label_income'].iloc[iterations * self.args.batch_size:
                                                                (iterations + 1) * self.args.batch_size].to_numpy()
                label_marital = test_labels['label_marital'].iloc[iterations * self.args.batch_size:
                                                                  (iterations + 1) * self.args.batch_size].to_numpy()

                iterations += 1

                total_income_auc += round(roc_auc_score(label_income, predictions[0]), 4)
                total_marital_stat_auc += round(roc_auc_score(label_marital, predictions[1]), 4)
            print("Iteration time: %0.4f ms" % elapsed_time)
            print("Processed %d records. (Test Income AUC, Test Marital Status AUC) = (%0.4f, %0.4f)"
                  % (num_processed_records, total_income_auc / iterations, total_marital_stat_auc / iterations))

    def validate_args(self):
        """validate the arguments"""

        if not self.args.data_location:
            if self.args.accuracy_only:
                raise ValueError("You must use real data for accuracy measurement.")


if __name__ == "__main__":
    evaluate_opt_graph = eval_classifier_optimized_graph()
    evaluate_opt_graph.run()
