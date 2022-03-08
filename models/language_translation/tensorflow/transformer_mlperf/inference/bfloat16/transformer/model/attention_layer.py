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
"""Implementation of multiheaded attention and self-attention layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from mlperf_compliance import mlperf_log


class Attention(tf.compat.v1.layers.Layer):
  """Multi-headed attention layer."""

  # set some variables to be static so that it can be computed only once, use 1.0 as initial
  # value
  rsqrtQ = 1.0
  depth = 1

  def __init__(self, hidden_size, num_heads, attention_dropout, train):
    if hidden_size % num_heads != 0:
      raise ValueError("Hidden size must be evenly divisible by the number of "
                       "heads.")

    super(Attention, self).__init__()
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.attention_dropout = attention_dropout
    self.train = train

    # Layers for linearly projecting the queries, keys, and values.
    use_bias = False
    self.q_dense_layer = tf.compat.v1.layers.Dense(hidden_size, use_bias=use_bias, name="q")
    self.k_dense_layer = tf.compat.v1.layers.Dense(hidden_size, use_bias=use_bias, name="k")
    self.v_dense_layer = tf.compat.v1.layers.Dense(hidden_size, use_bias=use_bias, name="v")
    self.output_dense_layer = tf.compat.v1.layers.Dense(hidden_size, use_bias=use_bias,
                                              name="output_transform")
    
    mlperf_log.transformer_print(
        mlperf_log.MODEL_HP_ATTENTION_DENSE,
        value={
          "hidden_size": hidden_size,
          "use_bias": use_bias,
          "num_heads": num_heads
        })

    # Scale q to prevent the dot product between q and k from growing too large.
    Attention.depth = (self.hidden_size // self.num_heads)
    Attention.rsqrtQ = Attention.depth ** -0.5

  def split_heads(self, x):
    """Split x into different heads, and transpose the resulting value.

    The tensor is transposed to insure the inner dimensions hold the correct
    values during the matrix multiplication.

    Args:
      x: A tensor with shape [batch_size, length, hidden_size]

    Returns:
      A tensor with shape [batch_size, num_heads, length, hidden_size/num_heads]
    """
    with tf.compat.v1.name_scope("split_heads"):
      batch_size = tf.shape(input=x)[0]
      length = tf.shape(input=x)[1]

      # Split the last dimension
      x = tf.reshape(x, [batch_size, length, self.num_heads, Attention.depth])

      # Transpose the result
      return tf.transpose(a=x, perm=[0, 2, 1, 3])

  def combine_heads(self, x):
    """Combine tensor that has been split.

    Args:
      x: A tensor [batch_size, num_heads, length, hidden_size/num_heads]

    Returns:
      A tensor with shape [batch_size, length, hidden_size]
    """
    with tf.compat.v1.name_scope("combine_heads"):
      batch_size = tf.shape(input=x)[0]
      length = tf.shape(input=x)[2]
      x = tf.transpose(a=x, perm=[0, 2, 1, 3])  # --> [batch, length, num_heads, depth]
      return tf.reshape(x, [batch_size, length, self.hidden_size])

  def call(self, x, y, bias, cache=None, encdec_cache=None):
    """Apply attention mechanism to x and y.

    Args:
      x: a tensor with shape [batch_size, length_x, hidden_size]
      y: a tensor with shape [batch_size, length_y, hidden_size]
      bias: attention bias that will be added to the result of the dot product.
      cache: (Used during prediction) dictionary with tensors containing results
        of previous attentions. The dictionary must have the items:
            {"k": tensor with shape [batch_size, i, key_channels],
             "v": tensor with shape [batch_size, i, value_channels]}
        where i is the current decoded length.

    Returns:
      Attention layer output with shape [batch_size, length_x, hidden_size]
    """
    # Linearly project the query (q), key (k) and value (v) using different
    # learned projections. This is in preparation of splitting them into
    # multiple heads. Multi-head attention uses multiple queries, keys, and
    # values rather than regular attention (which uses a single q, k, v).
    with tf.compat.v1.tpu.bfloat16_scope():
        if x.dtype == tf.float32:
           x = tf.cast(x, tf.bfloat16)
        if y.dtype == tf.float32:
           y = tf.cast(y, tf.bfloat16)
        q = self.q_dense_layer(x)
        if encdec_cache is not None:
          k = encdec_cache["k"]
          v = encdec_cache["v"]
        else:
          k = self.k_dense_layer(y)
          v = self.v_dense_layer(y)

    if cache is not None:
      # Combine cached keys and values with new keys and values.
      k = tf.concat([cache["k"], k], axis=1)
      v = tf.concat([cache["v"], v], axis=1)

      # Update cache
      cache["k"] = k
      cache["v"] = v

    # Split q, k, v into heads.
    q = self.split_heads(q)
    k = self.split_heads(k)
    v = self.split_heads(v)

    # Scale q to prevent the dot product between q and k from growing too large.
    # Calculate dot product attention
    with tf.compat.v1.tpu.bfloat16_scope():
        bias = tf.cast(bias, tf.bfloat16)
        logits = tf.matmul(q, k, transpose_b=True)
        logits *= Attention.rsqrtQ
        logits += bias
        weights = tf.nn.softmax(logits, name="attention_weights")
        if self.train:
          mlperf_log.transformer_print(
              key=mlperf_log.MODEL_HP_ATTENTION_DROPOUT,
              value=self.attention_dropout)
          weights = tf.nn.dropout(weights, 1 - (1.0 - self.attention_dropout))
        attention_output = tf.matmul(weights, v)

        # Recombine heads --> [batch_size, length, hidden_size]
        attention_output = self.combine_heads(attention_output)

        # Run the combined outputs through another linear projection layer.
        attention_output = self.output_dense_layer(attention_output)
        return attention_output


class SelfAttention(Attention):
  """Multiheaded self-attention layer."""

  def call(self, x, bias, cache=None):
    return super(SelfAttention, self).call(x, x, bias, cache)
