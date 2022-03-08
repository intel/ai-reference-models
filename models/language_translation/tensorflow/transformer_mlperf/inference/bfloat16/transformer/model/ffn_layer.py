# Copyright 2018 MLBenchmark Group. All Rights Reserved.
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
"""Implementation of fully connected network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from mlperf_compliance import mlperf_log


class FeedFowardNetwork(tf.compat.v1.layers.Layer):
  """Fully connected feedforward network."""

  def __init__(self, hidden_size, filter_size, relu_dropout, train):
    super(FeedFowardNetwork, self).__init__()
    self.hidden_size = hidden_size
    self.filter_size = filter_size
    self.relu_dropout = relu_dropout
    self.train = train

    use_bias = True
    self.filter_dense_layer = tf.compat.v1.layers.Dense(
        filter_size, use_bias=use_bias, activation=tf.nn.relu, name="filter_layer")
    self.output_dense_layer = tf.compat.v1.layers.Dense(
        hidden_size, use_bias=use_bias, name="output_layer")

    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_FFN_FILTER_DENSE,
        value={
          "filter_size": filter_size,
          "use_bias": use_bias,
          "activation": mlperf_log.RELU
        })
    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_FFN_OUTPUT_DENSE,
        value={
          "hidden_size": hidden_size,
          "use_bias": use_bias
        })

  def call(self, x, padding=None):
    # Retrieve dynamically known shapes
    batch_size = tf.shape(input=x)[0]
    length = tf.shape(input=x)[1]

    with tf.compat.v1.tpu.bfloat16_scope():
       # Reshape to 2D teansor
       x = tf.reshape(x, [-1, self.hidden_size])
       output = self.filter_dense_layer(x)
       if self.train:
         mlperf_log.transformer_print(
             key=mlperf_log.MODEL_HP_RELU_DROPOUT, value=self.relu_dropout)
         output = tf.nn.dropout(output, 1 - (1.0 - self.relu_dropout))
       output = self.output_dense_layer(output)

       # Reshaped back to 3D tensor
       output = tf.reshape(output, [batch_size, length, self.hidden_size])

    return output

