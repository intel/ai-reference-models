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
# ==============================================================================

import os
import tensorflow as tf
import tensorflow_gnn as tfgnn


def _preprocess_node_features(node_set, *, node_set_name):
    if node_set_name == "paper":
        # Retain the word2vec embedding unchanged.
        return {"embedding": node_set["embedding"]}
    elif node_set_name == "author":
        return {"hashed_id": tf.keras.layers.Hashing(num_bins=100_000)(
            node_set["index"])}
    elif node_set_name == "field_of_study":
        # Convert the string id to an index into an embedding table.
        return {"hashed_id": tf.keras.layers.Hashing(num_bins=50_000)(
            node_set["index"])}
    elif node_set_name == "institution":
        # Convert the string id to an index into an embedding table.
        return {"hashed_id": tf.keras.layers.Hashing(num_bins=6_500)(
            node_set["index"])}
    else:
        raise KeyError(f"Unexpected node_set_name='{node_set_name}'")


def _drop_all_features(graph_piece, **unused_kwargs):
    return {}


def _make_preprocessing_model(graph_tensor_spec, train_embed):
    batched_example_proto = tf.keras.layers.Input(
        shape=[], dtype=tf.string, name="examples")  # Name seen in SignatureDef.
    graph = tfgnn.keras.layers.ParseExample(graph_tensor_spec)(
        batched_example_proto)
    graph = graph.merge_batch_to_components()

    # Read the label for each subgraph off its root node.
    # By convention, the root node has index 0 in its node set,
    # but RaggedTensor doesn't let us subscript with [:, 0],
    # so we convert to a dense Tensor first.
    # If the labels feature is missing in the input, parsing returns
    # an empty RaggedTensor. The conversion to Tensor still works and
    # fills in a default value (clearly not a legal class id).
    labels = graph.context.features["label"]

    if train_embed:
        graph = tfgnn.keras.layers.MapFeatures(
            node_sets_fn=_preprocess_node_features,
            edge_sets_fn=_drop_all_features)(graph)
    # assert "labels" not in graph.node_sets["paper"].features
    return tf.keras.Model(batched_example_proto, (graph, labels))


def generate_dataset(graph_spec, data_location, batch_size):
    datasets = dict()
    for split in ["train", "valid", "test"]:
        data_dir = os.path.join(data_location, split)
        data_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir)]
        raw_dataset = tf.data.TFRecordDataset(data_files)
        dataset = raw_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        preproc_model = _make_preprocessing_model(graph_spec, True)
        dataset = dataset.map(preproc_model, deterministic=False, num_parallel_calls=tf.data.AUTOTUNE)
        datasets[split] = dataset
    return datasets
