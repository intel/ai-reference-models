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

'''Distilbert base inference, implementation adapted from Hugging Face Library https://huggingface.co/'''
import time, sys
import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import saved_model_pb2
from argparse import ArgumentParser
from transformers import AutoTokenizer
from datasets import load_from_disk
import threading

MAX_STEPS = 872
MAX_WARMUP_STEPS = 22

class distilbert_base_inference:

    def __init__(self):

        arg_parser = ArgumentParser(description="Distilbert int8 inference")
        arg_parser.add_argument("--task-name",
                                type=str,
                                help= "Name of the task to run benchmark, default: sst2",
                                default= "sst2",
                                dest="task_name"
                                )
        arg_parser.add_argument("-g", "--in-graph", help="Full path to the input graph ",
                                dest="input_graph", default=None) # TODO type=check_valid_filename)
        arg_parser.add_argument('-e', "--num-inter-threads",
                                help='The number of inter-thread.',
                                dest='num_inter_threads', type=int, default=0)
        arg_parser.add_argument('-a', "--num-intra-threads",
                                help='The number of intra-thread.',
                                dest='num_intra_threads', type=int, default=0)
        arg_parser.add_argument("--pad-to-max-length",
                                type= bool,
                                help= "Padding option",
                                default= True,
                                dest= "pad_to_max_length"
                                )
        arg_parser.add_argument("--warmup-steps", type=int, default=10,
                                help="number of warmup steps",
                                dest="warmup_steps")
        arg_parser.add_argument("--max-seq-length", type=int, default=128,
                                help="maximum total sequence length after Tokenization",
                                dest="max_seq_length")
        arg_parser.add_argument("--steps", type=int, default=850,
                                help="number of steps")
        arg_parser.add_argument("--data-location", type=str,
                                help="path to saved dataset",
                                dest="data_location")
        arg_parser.add_argument("--batch-size",
                                type= int,
                                default=32,
                                dest= "batch_size")
        arg_parser.add_argument("--accuracy-only", action="store_true", default=None)
        arg_parser.add_argument("--benchmark-only", action="store_true", default=None)
        self.args = arg_parser.parse_args()
        self.validate_args()

    def validate_args(self):
        if self.args.warmup_steps > MAX_WARMUP_STEPS:
            print("Warmup steps greater than max possible value of 22." + \
                  " Setting to max value of ", MAX_WARMUP_STEPS)
            self.args.warmup_steps = MAX_WARMUP_STEPS
        if self.args.accuracy_only:
            raise SystemExit("Accuracy is not run with weight sharing enabled")
        elif self.args.benchmark_only:
            if self.args.steps > (MAX_STEPS - MAX_WARMUP_STEPS):
                print("Steps greater than max possible value of {}.".format(MAX_STEPS - MAX_WARMUP_STEPS))
                print("Setting to max value of {}".format(MAX_STEPS - MAX_WARMUP_STEPS))
                self.args.steps = MAX_STEPS - MAX_WARMUP_STEPS
        if not self.args.data_location:
            raise SystemExit("Missing dataset path")

    def preprocess_function(self, examples):
        '''
            Tokenize the texts
        '''
        sentence1_key, sentence2_key = "sentence", None
        args = (
            (examples[sentence1_key],) if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = self.tokenizer(*args, padding="max_length",
                                max_length=self.args.max_seq_length,
                                truncation=True
        )
        return result

    def create_feed_dict_and_labels_benchmarking(self, start_idx, end_idx):
        ''' Return the input dictionary for the given batch '''
        if start_idx >= 0 and end_idx < 872:
            input_ids = np.array(self.dataset['input_ids'])[start_idx:end_idx, :]
            attention_mask = np.array(self.dataset['attention_mask'])[start_idx:end_idx,:]
            feed_dict = { 'input_ids:0': input_ids, 'attention_mask:0': attention_mask, }
            labels = np.array(self.dataset["label"])[start_idx:end_idx]
        return feed_dict, labels

    def run(self):
        '''Run benchmark for distilbert'''
        print("Run INT8 inference")
        # Load dataset (only validation split for inference)
        self.dataset = load_from_disk(self.args.data_location)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
                            "distilbert-base-uncased-finetuned-sst-2-english",
        )
        # Set max sequence length
        if self.args.max_seq_length > self.tokenizer.model_max_length:
            print(f'The max sequence length passed ({self.args.max_seq_length}) \
                   is larger than the max supported by model \
                   ({self.tokenizer.model_max_length}).Using max_seq_length =  \
                   {self.tokenizer.model_max_length}')
        self.args.max_seq_length = min(self.args.max_seq_length, self.tokenizer.model_max_length)

        # Tokenize the dataset
        self.dataset = self.dataset.map(self.preprocess_function, batched=True)

        # Load the frozen model
        graph = tf.Graph()
        with graph.as_default():
            graph_def = tf.compat.v1.GraphDef()
            with open(self.args.input_graph, "rb") as f:
                graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")

        output = graph.get_tensor_by_name('Identity:0')
        n_batch = int(self.args.steps / self.args.batch_size)
        if self.args.steps % self.args.batch_size != 0:
            n_batch += 1

        # Set the config for run
        config = tf.compat.v1.ConfigProto()
        config.intra_op_parallelism_threads=self.args.num_intra_threads
        config.inter_op_parallelism_threads=self.args.num_inter_threads
        infer_sess = tf.compat.v1.Session(graph=graph, config=config)

        throughput_list = []
        def run_model(sess, tid, feed_dict):
            time_consume = 0
            batch_size = self.args.batch_size
            warmup_steps = self.args.warmup_steps
            steps = self.args.steps

            # Warm up
            total_time = 0
            print("Started warmup for {} steps...".format(warmup_steps))
            start_step_idx = MAX_STEPS - MAX_WARMUP_STEPS
            for step in range(start_step_idx, start_step_idx + warmup_steps):
                _ = sess.run(output, feed_dict= feed_dict)
            print("Warmup completed.")

            # Inference
            print("Starting inference for {} steps...".format(steps))
            total_correct_predictions = 0
            for batch_id in range(n_batch):
                start = time.time()
                pred = sess.run(output, feed_dict=feed_dict)
                end = time.time()
                time_consume = end - start
                total_time += time_consume
            time_per_batch = total_time / float(steps / batch_size)
            throughput = batch_size / time_per_batch
            throughput_list.append(throughput)

            if self.args.benchmark_only:
                if self.args.batch_size == 1:
                    print('Latency: {:.4f} ms'.format(time_per_batch * 1000))
                print('Throughput: {:.4f} sentences/sec'.format(throughput))

        threads = []
        num_instances = self.args.num_intra_threads//4
        step_idx = MAX_STEPS - MAX_WARMUP_STEPS
        start_idx = 0
        end_idx = self.args.batch_size
        feed_dict, _ = self.create_feed_dict_and_labels_benchmarking(start_idx, end_idx)
        for i in range(1, num_instances+1):
            thread = threading.Thread(target=run_model, args=(infer_sess, i, feed_dict))
            threads.append(thread)
            thread.start()

        for index, thread in enumerate(threads):
            thread.join()

        total_throughput = 0
        for i in range(0, num_instances):
            total_throughput += throughput_list[i]
        print('Total aggregated Throughput %f' %(total_throughput))

if __name__=="__main__":
    distilbert_ob = distilbert_base_inference()
    distilbert_ob.run()
