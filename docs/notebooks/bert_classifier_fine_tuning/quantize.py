#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
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

import argparse
import csv
import numpy as np
import os

from neural_compressor.experimental import common, Quantization

import tokenization
from run_classifier import InputExample
from run_classifier import convert_examples_to_features


def get_args():
    """ Parse args that are used for quantization """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-yaml",
                        default="imbd.yaml",
                        type=str,
                        help="Neural compressor config yaml file")
    parser.add_argument("--vocab-file",
                        default="vocab.txt",
                        type=str,
                        help="Path to the vocab.txt file")
    parser.add_argument("--tsv-dir",
                        default="imdb_tsv",
                        type=str,
                        help="Path to the directory that has IMDB .tsv files")
    parser.add_argument("--output-dir",
                        default=None,
                        type=str,
                        help="Path to the output directory from training where the saved model")
    parser.add_argument("--max-seq-length",
                        default=128,
                        type=int,
                        help="Max sequence length")
    parser.add_argument("--batch-size",
                        default=8,
                        type=int,
                        help="Batch size")

    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args):
    """ Validate args to verify that various paths exist """
    if not os.path.exists(args.config_yaml):
        raise ValueError("The specified config yaml ({}) does not exist".format(args.config_yaml))

    if not os.path.exists(args.vocab_file):
        raise ValueError("The specified vocab file ({}) does not exist.".format(args.vocab_file))

    if not os.path.isdir(args.tsv_dir):
        raise ValueError("The specified tsv dir ({}) does not exist.".format(args.tsv_dir))

    if not os.path.isdir(args.output_dir):
        raise ValueError("The specified output directory ({}) does not exist.".format(args.output_dir))


def create_examples(lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
        # Only the test set has a header
        if set_type == "test" and i == 0:
            continue
        guid = "%s-%s" % (set_type, i)
        if set_type == "test":
            text_a = tokenization.convert_to_unicode(line[1])
            label = "0"
        else:
            text_a = tokenization.convert_to_unicode(line[3])
            label = tokenization.convert_to_unicode(line[1])
        examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples


def read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
        return lines


def get_dev_examples(data_dir):
    """See creates examples from the dev.tsv file. """
    return create_examples(read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")


class Accuracy(object):
    def __init__(self):
        self.pred_list = []
        self.label_list = []
        self.num_samples = 0

    def update(self, preds, label):
        self.pred_list.extend(np.argmax(preds[0], axis=1))
        self.label_list.extend(list(label))
        self.num_samples += len(label)

    def reset(self):
        self.pred_list = []
        self.label_list = []
        self.num_samples = 0

    def result(self):
        correct_num = np.sum(np.array(self.pred_list) == np.array(self.label_list))
        return correct_num / self.num_samples


class TF_BERTDataSet():
    def __init__(self, data_dir, tokenizer, max_seq_length):
        eval_examples = get_dev_examples(data_dir)
        self.max_seq_length = max_seq_length
        self.features = convert_examples_to_features(
            examples=eval_examples,
            label_list=["0", "1"],
            max_seq_length=max_seq_length,
            tokenizer=tokenizer)
        print(self.features)

    def __getitem__(self, idx):
        input_ids_data = self.features[idx].input_ids
        input_mask_data = self.features[idx].input_mask
        segment_ids_data = self.features[idx].segment_ids
        if len(input_ids_data) < self.max_seq_length:
            input_ids_data += [0] * (self.max_seq_length - len(input_ids_data))
            input_mask_data += [0] * (self.max_seq_length - len(input_mask_data))
            segment_ids_data += [0] * (self.max_seq_length - len(segment_ids_data))
        return (np.array(input_mask_data).astype('int32'),
                np.array(input_ids_data).astype('int32'),
                self.features[idx].label_id,
                np.array(segment_ids_data).astype('int32')), self.features[idx].label_id

    def __len__(self):
        return len(self.features)


def main():
    args = get_args()

    vocab_file = args.vocab_file
    tsv_dir = args.tsv_dir
    max_seq_length = args.max_seq_length
    batch_size = args.batch_size
    output_dir = args.output_dir

    quantizer = Quantization(args.config_yaml)

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
    dataset = TF_BERTDataSet(tsv_dir, tokenizer, max_seq_length)

    quantizer.model = common.Model(os.path.join(output_dir, "frozen"))
    quantizer.calib_dataloader = common.DataLoader(dataset, batch_size)
    quantizer.eval_dataloader = common.DataLoader(dataset, batch_size)
    quantizer.metric = common.Metric(metric_cls=Accuracy)

    q_model = quantizer()

    q_model.save(os.path.join(output_dir, "quantized"))


if __name__ == '__main__':
    main()
