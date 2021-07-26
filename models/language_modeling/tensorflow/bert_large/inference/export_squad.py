#! /usr/bin/env python

# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from absl import app
from absl import logging
tf.compat.v1.disable_v2_behavior()

tf.compat.v1.flags.DEFINE_bool("saved_model",
                               False,
                               "whether export saved model or not")
FLAGS = tf.compat.v1.flags.FLAGS

# We just import classifier here for `create_model` and some processors such as
# MNLI or MRPC. Because of the flags defined in `run_classifier.py`, we need not
# to define the flags again.
from run_squad import create_model_top
from modeling import BertConfig

class SquadExporter:
  def __init__(self,
               output_dir: str,
               bert_config: str,
               max_seq_length: int):

    bert_config = BertConfig.from_json_file(bert_config)
    bert_config.experimental_gelu = FLAGS.experimental_gelu
    if FLAGS.precision:
      bert_config.precision = FLAGS.precision

    self.session = tf.compat.v1.Session()

    placeholder = tf.compat.v1.placeholder
    input_shape = [None, max_seq_length]
    self.input_ids = placeholder(tf.int32, input_shape, name='input_ids')
    self.input_mask = placeholder(tf.int32, input_shape, name='input_mask')
    self.segment_ids = placeholder(tf.int32, input_shape, name='segment_ids')

    (self.start_logits, self.end_logits) = \
      create_model_top(bert_config, False, # is training
                       self.input_ids, self.input_mask, self.segment_ids,
                       False, # use_one_hot_embeddings
                       None) # frozen graph path

    latest_model = tf.train.latest_checkpoint(FLAGS.output_dir)
    saver = tf.compat.v1.train.Saver()
    saver.restore(self.session, latest_model)

    self.output_dir = output_dir
    self.dest_dir = os.path.join(self.output_dir, "frozen")
    if not os.path.exists(self.dest_dir):
      os.mkdir(self.dest_dir)

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, exc_tb):
    self.session.close()

  def export(self, saved_model: bool):
    if saved_model:
      self.export_saved_model()

    self.export_frozen_graph()

  def export_saved_model(self,
                         signature_def_name="eval",
                         tag=tf.compat.v1.saved_model.tag_constants.SERVING):
    build_tensor_info = tf.compat.v1.saved_model.build_tensor_info
    signature_def_utils = tf.compat.v1.saved_model.signature_def_utils
    inputs = {
        'input_ids': build_tensor_info(self.input_ids),
        'input_mask': build_tensor_info(self.input_mask),
        'segment_ids': build_tensor_info(self.segment_ids)
    }

    outputs = {
        "start_logits": build_tensor_info(self.start_logits),
        "end_logits": build_tensor_info(self.end_logits),
    }

    signature = signature_def_utils.build_signature_def(inputs, outputs)
    signature_def_map = {signature_def_name: signature}

    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(self.dest_dir)
    builder.add_meta_graph_and_variables(self.session, [tag], signature_def_map)
    builder.save()

  def export_frozen_graph(self, frozen_graph_name="frozen_graph.pb"):
    # we should disable v2 behavior, at the same time, the bn norm has some op name difference
    # should be handled. Otherwise, it will throw exception when do import graph def.
    # https://www.bountysource.com/issues/36614355-unable-to-import-frozen-graph-with-batchnorm
    graph_def = self.session.graph.as_graph_def()
    for node in graph_def.node:
      if node.op == 'RefEnter':
        node.op = 'Enter'
        for index in range(len(node.input)):
          if 'moving_' in node.input[index]:
            node.input[index] = node.input[index] + '/read'
      if node.op == 'RefSwitch':
        node.op = 'Switch'
        for index in range(len(node.input)):
          if 'moving_' in node.input[index]:
            node.input[index] = node.input[index] + '/read'
      elif node.op == 'AssignSub':
        node.op = 'Sub'
        if 'use_locking' in node.attr: del node.attr['use_locking']
      elif node.op == 'AssignAdd':
        node.op = 'Add'
        if 'use_locking' in node.attr: del node.attr['use_locking']

    outputs_name = ['start_logits', 'end_logits']
    graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(self.session,
                                                                       graph_def,
                                                                       outputs_name)

    path = os.path.join(self.dest_dir, frozen_graph_name)
    with tf.compat.v1.gfile.GFile(path, 'wb') as pb_file:
      pb_file.write(graph_def.SerializeToString())

def main(_):
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

  with SquadExporter(FLAGS.output_dir,
                     FLAGS.bert_config_file,
                     FLAGS.max_seq_length) as exporter:
    exporter.export(FLAGS.saved_model)

if __name__ == "__main__":
  tf.compat.v1.flags.mark_flag_as_required("bert_config_file")
  tf.compat.v1.flags.mark_flag_as_required("output_dir")
  tf.compat.v1.app.run()
