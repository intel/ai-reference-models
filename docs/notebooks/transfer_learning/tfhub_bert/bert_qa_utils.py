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

import glob
import json
import os
import pandas as pd
import tensorflow as tf
import wget

from official.common import distribute_utils
from official.nlp.bert.run_squad_helper import get_dataset_fn

from zipfile import ZipFile


def create_mini_dataset_file(original_file, output_file, num_dataset_items, overwrite=False):
    """
    Creates a mini version of the specified json file. The original_file is expected to be in a format
    similar to the SQuAD dataset. The number of dataset items represents the number of child elements
    under the "data" tag that will be grabbed for the mini dataset. Dataset items will be randomly
    selected from the original dataset. Each child element may contain several sets of articles with
    questions/answers. The overwrite flag specifies whether or not to overwrite a mini dataset file
    that already exists. If overwrite=False and the mini dataset file already exists, nothing will happen.
    """
    if not os.path.exists(output_file) or overwrite:
        import random
        
        with open(original_file) as f:
            original_data = json.load(f)

        total_len = len(original_data["data"])
        
        if num_dataset_items > total_len:
            raise ValueError("The number of dataset items ({}) cannot be more than the total "
                             "dataset length ({}).".format(num_dataset_items, total_len))
        
        item_indicies = random.sample(range(0, total_len), num_dataset_items)
        print("Total dataset length:", total_len)
        print("Randomly selected dataset indices:", item_indicies)
        
        articles = []
        
        for data_index in item_indicies:
            article = {}
            article["paragraphs"] = original_data["data"][data_index]["paragraphs"]
            article["title"] = original_data["data"][data_index]["title"]
            
            for p in article["paragraphs"]:
                for qas in p["qas"]:
                    qas["id"] = str(qas["id"])
            
            articles.append(article)

        # Add the article to a dictionary for the mini dataset
        mini_data = {}
        mini_data["data"] = articles
        
        # Add on a version
        mini_data["version"] = original_data["version"] if "version" in original_data.keys() else "1.0"

        with open(output_file, "w") as f:
            f.write(json.dumps(mini_data, indent=4))

        if os.path.exists(output_file):
            print("Wrote dataset file with {} articles to: {}".format(num_dataset_items, output_file))
    else:
        print("Found existing dataset file:", output_file)


def display_predictions(predict_data_path, results_file_path, n=10):
    """ Displays n number of predictions along with the actual value """
    
    def get_data_list():
        count = 0
        data_list = []
        with open(predict_data_path, "r") as actual_data_file:
            actual_data = json.load(actual_data_file)["data"]
            with open(results_file_path, "r") as results_file:
                results = json.load(results_file)
                for actual_item in actual_data:
                    for actual_paragraph in actual_item["paragraphs"]:
                        for actual_qas in actual_paragraph["qas"]:
                            if "is_impossible" in actual_qas.keys() and actual_qas["is_impossible"]:
                                actual_answer = "is_impossible"
                            elif len(actual_qas["answers"]) >= 1:
                                answers_text = [x["text"] for x in actual_qas["answers"]]
                                actual_answer = "<br>".join(set(answers_text))
                            else:
                                actual_answer = "Unknown"
                            question = actual_qas["question"]
                            prediction = results[actual_qas["id"]]
                            data_list.append([question, prediction, actual_answer])
                            count += 1
                            if count > n:
                                return data_list

    predict_df = pd.DataFrame(get_data_list(),
                              columns=["Question",
                                       "Predicted Answer",
                                       "Actual Answer(s)"])
    return predict_df.style.hide(axis="index")
                    

def get_config_and_vocab_from_zip(zip_url, bert_dir):
    """
    We are loading the trained BERT model from TF Hub, however the run_squad.py scripts still
    require us to pass in a vocab.txt and bert config file. We can get these from the checkpoint
    .zip files. The directory structure of the .zip files for each BERT model is not
    consistent, so there's a glob search being done to locate the actual vocab.txt and
    bert_config.json file after they're extracted from the zip (sometimes they are in
    subdirectories).
    :param zip_url: URL where the checkpoint zip can be downloaded
    :param bert_dir: BERT directory where the vocab.txt and bert_config.json should be copied
    :return: Paths to the vocab.txt and bert_config.json
    """
    vocab_txt = os.path.join(bert_dir, "vocab.txt")
    bert_config = os.path.join(bert_dir, "bert_config.json")
    
    if not os.path.exists(vocab_txt) or not os.path.exists(bert_config):
        downloaded_file = wget.download(zip_url, bert_dir)
        with ZipFile(downloaded_file, "r") as checkpoint_zip:
            def get_file_from_zip(file_path):
                file_basename = os.path.basename(file_path)
                for zipinfo in checkpoint_zip.infolist():
                    if file_basename in zipinfo.filename:
                        checkpoint_zip.extract(member=zipinfo.filename, path=bert_dir)

                        if not os.path.exists(file_path):
                            # the file isn't directly in the bert_dir, so search subfolders and move it
                            search_path = os.path.join(bert_dir, "**", file_basename)
                            matches = glob.glob(search_path, recursive=True)
                            if matches:
                                os.replace(matches[0], file_path)
                        break
            
            if not os.path.exists(vocab_txt):
                get_file_from_zip(vocab_txt)
            
            if not os.path.exists(bert_config):
                get_file_from_zip(bert_config)

        os.remove(downloaded_file)
        
    return vocab_txt, bert_config


# This function was taken from the TensorFlow Model Garden repo and adapted
# to be a utility function that has a string for the strategy, directly passes
# in the max_seq_length instead of a metadata object, and removes the need for FLAGS
# being defined (instead just passes in the predict_batch_size as an arg).

# https://github.com/tensorflow/models/blob/v2.7.0/official/nlp/bert/run_squad_helper.py#L176
def predict_squad_customized(strategy_str, max_seq_length, predict_batch_size,
                             predict_tfrecord_path, num_steps, squad_model):
    
    strategy = distribute_utils.get_distribution_strategy(distribution_strategy=strategy_str)
    
    """Make predictions using a Bert-based squad model."""
    predict_dataset_fn = get_dataset_fn(
        predict_tfrecord_path,
        max_seq_length,
        predict_batch_size,
        is_training=False)
    predict_iterator = iter(
        strategy.distribute_datasets_from_function(predict_dataset_fn))

    @tf.function
    def predict_step(iterator):
        """Predicts on distributed devices."""

        def _replicated_step(inputs):
            """Replicated prediction calculation."""
            x, _ = inputs
            unique_ids = x.pop('unique_ids')
            start_logits, end_logits = squad_model(x, training=False)
            return dict(
                unique_ids=unique_ids,
                start_logits=start_logits,
                end_logits=end_logits)

        outputs = strategy.run(_replicated_step, args=(next(iterator),))
        return tf.nest.map_structure(strategy.experimental_local_results, outputs)

    all_results = []
    for _ in range(num_steps):
        predictions = predict_step(predict_iterator)
        for result in get_raw_results(predictions):
            all_results.append(result)
        if len(all_results) % 100 == 0:
            print('Made predictions for %d records.', len(all_results))
    return all_results
