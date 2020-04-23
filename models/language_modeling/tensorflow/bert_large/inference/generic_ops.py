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
"""The main BERT model and related functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


_inprecision = tf.float32
_rprecision = tf.float32
_use_mkldnn = True 

def set_rprecision(dt):
  global _rprecision
  _rprecision=dt

def set_mkldnn(mkldnn=False):
  global _use_mkldnn
  _use_mkldnn = mkldnn

def i_cast(x) :
     return tf.cast(x, _inprecision)

def r_cast(x) :
     return tf.cast(x, _rprecision)

def multiply(x,y):
    x = r_cast(x)
    y = r_cast(y)
    return tf.multiply(x,y)

def mzip(x,y):
    if x.dtype== tf.bfloat16:
      x = r_cast(x)
      y = r_cast(y)
    return zip(x,y)

def tanh(x):
    x = i_cast(x)
    rval = tf.tanh(x)
    return r_cast(rval)

def softmax(scores, axis=None):
    if _use_mkldnn:
      #print("softmax using mkldnn")  
      return tf.nn.softmax(scores, axis)
    else:
      scores = i_cast(scores)
      rval = tf.nn.softmax(scores, axis)
      return r_cast(rval)

def layer_norm(inputs, begin_norm_axis, begin_params_axis, scope):
    type_string = 'float32' if _rprecision == tf.float32 else 'mixed_bfloat16'
    policy = tf.keras.mixed_precision.experimental.Policy(type_string)
    lnorm = tf.keras.layers.LayerNormalization(dtype=policy)
    return lnorm(inputs)

"Moved from modeling.py"
def gelu(x):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.

  Returns:
    `x` with the GELU activation applied.
  """
  if _use_mkldnn:
    #print("gelu using mkldnn")  
    return tf.nn.gelu(x)
  else:
    x = i_cast(x)
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    rval = x * cdf
    return r_cast(rval)

def logTheLossHook(total_loss, n):
    return tf.compat.v1.train.LoggingTensorHook({"\t Loss " : total_loss}, every_n_iter=n)

