#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
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

import time
import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from argparse import ArgumentParser
from transformers import AutoTokenizer
from datasets import load_from_disk

DATASET_LEN = 872

class distilbert_base_inference:

    def __init__(self):

        arg_parser = ArgumentParser(description="Distilbert fp32 inference")
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
        arg_parser.add_argument("--steps", type=int, default=50,
                                help="number of steps")
        arg_parser.add_argument("--data-location", type=str,
                                help="path to saved dataset",
                                dest="data_location")
        arg_parser.add_argument("--batch-size",
                                type= int,
                                default=32,
                                dest= "batch_size")
        arg_parser.add_argument('-p', "--precision", type=str, default='fp32',
                                help="precision/datatype")
        arg_parser.add_argument("--accuracy-only", action="store_true", default=None)
        arg_parser.add_argument("--benchmark-only", action="store_true", default=None)
        self.args = arg_parser.parse_args()
        self.validate_args()

    def validate_args(self):
        if not self.args.data_location:
            raise SystemExit("Missing dataset path")

    def get_correct_predictions(self, preds, label_ids):
        '''
            Evaluate the predictions and return the total
            number of correct predictions
        '''
        preds = np.argmax(preds, axis=1)
        correct_preds = 0
        for pred, label in zip(preds, label_ids):
            if pred == label:
                correct_preds += 1
        return correct_preds

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

    def create_feed_dict_and_labels(self, batch_id= None, n_batches= None, idx= None):
        '''
            Return the input dictionary for the given batch
        '''
        if idx is None:
            start_idx = batch_id * self.args.batch_size
            if batch_id == n_batches - 1:
                end_idx = DATASET_LEN
            else:
                end_idx = start_idx + self.args.batch_size
            input_ids = np.array(self.dataset['input_ids'])[start_idx:end_idx, :]
            attention_mask = np.array(self.dataset['attention_mask'])[start_idx:end_idx,:]
            feed_dict = { 'input_ids:0': input_ids,
                        'attention_mask:0': attention_mask,
            }
            labels = np.array(self.dataset["label"])[start_idx:end_idx]
        else:
            input_ids = np.array(self.dataset['input_ids'])[idx, :].reshape(1, -1)
            attention_mask = np.array(self.dataset['attention_mask'])[idx,:].reshape(1, -1)
            feed_dict = { 'input_ids:0': input_ids,
                        'attention_mask:0': attention_mask,
            }
            labels = np.array(self.dataset["label"])[idx]
        return feed_dict, labels

    def run(self):
        '''Run benchmark for distilbert'''

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

        # Set the config for run
        config = tf.compat.v1.ConfigProto()
        config.intra_op_parallelism_threads=self.args.num_intra_threads
        config.inter_op_parallelism_threads=self.args.num_inter_threads
        if self.args.precision == "bfloat16":
            config.graph_options.rewrite_options.auto_mixed_precision_onednn_bfloat16 = rewriter_config_pb2.RewriterConfig.ON
        elif self.args.precision == "fp16":
            config.graph_options.rewrite_options.auto_mixed_precision = rewriter_config_pb2.RewriterConfig.ON

        # Load the frozen model
        if self.args.precision == "int8":
            graph = tf.Graph()
            with graph.as_default():
                graph_def = tf.compat.v1.GraphDef()
                with open(self.args.input_graph, "rb") as f:
                    graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name="")
        else:
            sm = saved_model_pb2.SavedModel()
            with tf.io.gfile.GFile(self.args.input_graph, "rb") as f:
                sm.ParseFromString(f.read())
            g_def = sm.meta_graphs[0].graph_def
            with tf.Graph().as_default() as graph:
                tf.import_graph_def(g_def, name='')

        # Define the session
        output = graph.get_tensor_by_name('Identity:0')
        sess =  tf.compat.v1.Session(graph=graph, config=config)
        n_batch = int(DATASET_LEN / self.args.batch_size)
        if DATASET_LEN % self.args.batch_size != 0:
            n_batch += 1

        # Warm up
        total_time = 0
        feed_dict, _ = self.create_feed_dict_and_labels(1,1)
        print("Started warmup for {} steps...".format(self.args.warmup_steps))
        for _ in range(self.args.warmup_steps):
            _ = sess.run(output, feed_dict=feed_dict)
        print("Warmup completed.")

        # Inference
        print("Starting inference for SST2 Validation dataset...")
        total_correct_predictions = 0
        for batch_id in range(n_batch):
            feed_dict, labels = self.create_feed_dict_and_labels(batch_id, n_batch)
            start_time = time.time()
            pred = sess.run(output, feed_dict=feed_dict)
            run_time = time.time() - start_time
            total_correct_predictions += self.get_correct_predictions(pred, labels)
            total_time += run_time
        time_per_batch = total_time / float(DATASET_LEN / self.args.batch_size)

        accuracy = total_correct_predictions / DATASET_LEN
        print('Accuracy: {:.4f}'.format(accuracy))
        print('Throughput: {:.4f} sentences/sec'.format(self.args.batch_size / time_per_batch))

if __name__=="__main__":
    distilbert_ob = distilbert_base_inference()
    distilbert_ob.run()

