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
from tensorflow.python.platform import tf_logging
import numpy as np
from tensorflow.core.protobuf import rewriter_config_pb2
import utils
import dataloader
import os

np.random.seed(123)


class eval_graphsage:
    """Evaluate GraphSAGE model"""

    def __init__(self):

        arg_parser = ArgumentParser(description="Parse args")

        arg_parser.add_argument(
            "-b",
            "--batch-size",
            help="Specify the batch size. If this "
            "parameter is not specified, then "
            "it will run with batch size of 1000 ",
            dest="batch_size",
            type=int,
            default=1000,
        )

        arg_parser.add_argument(
            "-p",
            "--precision",
            help="Specify the model precision to use: fp32, bfloat16, fp16 or int8",
            required=True,
            choices=["fp32", "bfloat16", "fp16", "int8"],
            dest="precision",
        )

        arg_parser.add_argument(
            "-e",
            "--num-inter-threads",
            help="The number of inter-thread.",
            dest="num_inter_threads",
            type=int,
            default=0,
        )

        arg_parser.add_argument(
            "-a",
            "--num-intra-threads",
            help="The number of intra-thread.",
            dest="num_intra_threads",
            type=int,
            default=0,
        )

        arg_parser.add_argument(
            "-m",
            "--pretrained-model",
            help="Specify the path to the pretrained model",
            dest="pretrained_model",
            required=True,
        )

        arg_parser.add_argument(
            "-d",
            "--data-location",
            help="Specify the location of the data. ",
            dest="data_location",
            required=True,
        )

        arg_parser.add_argument(
            "-r",
            "--accuracy-only",
            help="For accuracy measurement only.",
            dest="accuracy_only",
            action="store_true",
        )

        arg_parser.add_argument(
            "-s", "--steps", type=int, default=5524, help="number of steps"
        )

        arg_parser.add_argument(
            "--warmup-steps", type=int, default=20, help="number of warmup steps"
        )

        # parse the arguments
        self.args = arg_parser.parse_args()
        # validate the arguements
        self.validate_args()

    def run(self):
        data_location = self.args.data_location
        pretrained_model = self.args.pretrained_model
        warmup_iter = self.args.warmup_steps
        steps = self.args.steps
        data = dataloader.load_data(prefix=data_location + "/ppi")
        G = data[0]
        features = data[1]
        id_map = data[2]
        class_map = data[4]
        if isinstance(list(class_map.values())[0], list):
            num_classes = len(list(class_map.values())[0])
        else:
            num_classes = len(set(class_map.values()))

        context_pairs = data[3]
        placeholders = utils.construct_placeholders(num_classes)
        minibatch = utils.NodeMinibatchIterator(
            G,
            id_map,
            placeholders,
            class_map,
            num_classes,
            batch_size=self.args.batch_size,
            context_pairs=context_pairs,
        )

        infer_config = tf.compat.v1.ConfigProto()
        infer_config.intra_op_parallelism_threads = self.args.num_intra_threads
        infer_config.inter_op_parallelism_threads = self.args.num_inter_threads
        infer_config.use_per_session_threads = 1

        tf_xla_enabled = False
        if "TF_XLA_FLAGS" in os.environ:
            tf_xla_flags = os.environ["TF_XLA_FLAGS"].split(sep=" ")
            tf_xla_enabled = (
                "--tf_xla_auto_jit=2" in tf_xla_flags
                and "--tf_xla_cpu_global_jit" in tf_xla_flags
            )

        if tf_xla_enabled:
            # Disable remapper fusions, so as to allow fusions via XLA
            infer_config.graph_options.rewrite_options.remapping = (
                rewriter_config_pb2.RewriterConfig.OFF
            )

        if self.args.precision == "bfloat16":
            print("Enabling auto-mixed precision for bfloat16")
            infer_config.graph_options.rewrite_options.auto_mixed_precision_onednn_bfloat16 = (
                rewriter_config_pb2.RewriterConfig.ON
            )
        elif self.args.precision == "fp16":
            print("Enabling auto-mixed precision for fp16")
            infer_config.graph_options.rewrite_options.auto_mixed_precision = (
                rewriter_config_pb2.RewriterConfig.ON
            )

        if self.args.precision == "int8":
            graph_def = tf.compat.v1.get_default_graph().as_graph_def()
            with tf.io.gfile.GFile(
                os.path.join(self.args.pretrained_model, "graphsage_int8.pb"), "rb"
            ) as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
        else:
            graph_def = tf.compat.v1.get_default_graph().as_graph_def()
            with tf.io.gfile.GFile(
                os.path.join(self.args.pretrained_model, "graphsage_frozen_model.pb"),
                "rb",
            ) as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())

        # Import the graph
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")

        sess = tf.compat.v1.Session(config=infer_config, graph=graph)
        output_tensor = sess.graph.get_tensor_by_name("Sigmoid:0")

        total_time = 0
        manual_step = False

        def infer(sess, size, output_tensor, minibatch, warmup_iter, test):
            t_test = time.time()
            val_losses = []
            val_preds = []
            labels = []
            iter_num = 0
            manual_step = False
            finished = False
            total_time = 0
            while not finished:
                (
                    feed_dict_val,
                    batch_labels,
                    finished,
                    _,
                ) = minibatch.incremental_node_val_feed_dict(size, iter_num, test=True)
                if iter_num == 0:
                    cur_warmup_step = 0
                    while cur_warmup_step < warmup_iter:
                        node_outs_val = sess.run(
                            [output_tensor], feed_dict=feed_dict_val
                        )
                        cur_warmup_step += 1
                if iter_num == steps:
                    manual_step = True
                    break
                tf_logging.warn("\n---> Start iteration {0}".format(str(iter_num)))
                start_time = time.time()
                node_outs_val = sess.run([output_tensor], feed_dict=feed_dict_val)
                time_consume = time.time() - start_time
                val_preds.append(node_outs_val[0])
                labels.append(batch_labels)
                total_time += time_consume
                tf_logging.warn("\n---> Stop iteration {0}".format(str(iter_num)))
                iter_num += 1
            time_average = total_time / iter_num
            if manual_step:
                return 0.0, (time.time() - t_test) / iter_num, time_average
            val_preds = np.vstack(val_preds)
            labels = np.vstack(labels)
            f1_scores = utils.calc_f1(labels, val_preds)
            return f1_scores, (time.time() - t_test) / iter_num, time_average

        test_f1_micro, duration, time_average = infer(
            sess, self.args.batch_size, output_tensor, minibatch, warmup_iter, test=True
        )
        if manual_step or (not self.args.accuracy_only):
            print("Benchmark")
            print("Precision ", self.args.precision)
            print("Average time: %.6f sec" % (time_average))
            print("Batch size = %d" % self.args.batch_size)
            if self.args.batch_size == 1:
                print("Latency: %.3f ms" % (time_average * 1000))
            print(
                "Throughput: %.3f examples/sec" % (self.args.batch_size / time_average)
            )
        else:
            print("Accuracy")
            print("Avg evaluation time: %0.4f ms" % time_average)
            print("Test accuracy: %0.4f " % (test_f1_micro))

    def validate_args(self):
        """validate the arguments"""

        if not self.args.data_location:
            if self.args.accuracy_only:
                raise ValueError("You must use real data for accuracy measurement.")


if __name__ == "__main__":
    eval_graphsage = eval_graphsage()
    eval_graphsage.run()
