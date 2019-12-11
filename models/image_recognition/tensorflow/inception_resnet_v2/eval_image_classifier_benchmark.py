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


# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import time
import numpy as np
from argparse import ArgumentParser
import tensorflow as tf


class EvalClassifierBenchmark:
    """Evaluate image classifier with int8 TensorFlow graph"""

    def __init__(self):

        arg_parser = ArgumentParser(description='Parse args')

        arg_parser.add_argument('-b', "--batch-size",
                                help="Specify the batch size. If this "
                                     "parameter is not specified or is -1, the "
                                     "largest ideal batch size for the model will "
                                     "be used.",
                                dest="batch_size", type=int, default=-1)

        arg_parser.add_argument('-e', "--inter-op-parallelism-threads",
                                help='The number of inter-thread.',
                                dest='num_inter_threads', type=int, default=0)

        arg_parser.add_argument('-a', "--intra-op-parallelism-threads",
                                help='The number of intra-thread.',
                                dest='num_intra_threads', type=int, default=0)

        arg_parser.add_argument('-g', "--input-graph",
                                help='Specify the input graph for the transform tool',
                                dest='input_graph')

        self.args = arg_parser.parse_args()

    def run(self):
        """run benchmark with optimized graph"""

        with tf.Graph().as_default() as graph:

            config = tf.ConfigProto()
            config.allow_soft_placement = True
            config.intra_op_parallelism_threads = self.args.num_intra_threads
            config.inter_op_parallelism_threads = self.args.num_inter_threads

            with tf.Session(config=config) as sess:
                # import the quantized graph
                with tf.gfile.FastGFile(self.args.input_graph, 'rb') as input_file:
                    graph_def = tf.GraphDef()
                    input_graph_content = input_file.read()
                    graph_def.ParseFromString(input_graph_content)

                    sess.graph.as_default()
                    tf.import_graph_def(graph_def, name='')

                    # Definite input and output Tensors for detection_graph
                    image = graph.get_tensor_by_name('input:0')
                    predict = graph.get_tensor_by_name('InceptionResnetV2/Logits/Predictions:0')
                    tf.global_variables_initializer()

                    i = 0
                    num_iteration = 40
                    warm_up_iteration = 10
                    total_time = 0
                    for _ in range(num_iteration):
                        i += 1
                        image_np = np.random.rand(self.args.batch_size, 299, 299, 3).astype(np.uint8)
                        start_time = time.time()
                        (predicts) = sess.run([predict], feed_dict={image: image_np})
                        time_consume = time.time() - start_time
                        print('Iteration %d: %.3f sec' % (i, time_consume))
                        if i > warm_up_iteration:
                            total_time += time_consume

                    time_average = total_time / (num_iteration - warm_up_iteration)
                    print('Average time: %.3f sec' % (time_average))

                    print('Batch size = %d' % self.args.batch_size)
                    if (self.args.batch_size == 1):
                        print('Latency: %.3f ms' % (time_average * 1000))
                    # print throughput for both batch size 1 and 128
                    print('Throughput: %.3f images/sec' % (self.args.batch_size / time_average))


if __name__ == "__main__":
    eval_benchmark = EvalClassifierBenchmark()
    eval_benchmark.run()
