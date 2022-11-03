# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A collection of JAX utility functions for use in protein folding."""

import collections
import numbers
from typing import Mapping
import jax
import jax.numpy as jnp
import numpy as np
import torch


def final_init(config):
  if config.zero_init:
    return 'zeros'
  else:
    return 'linear'


def batched_gather(params, indices, axis=0, batch_dims=0):
  """Implements a JAX equivalent of `tf.gather` with `axis` and `batch_dims`."""
  take_fn = lambda p, i: jnp.take(p, i, axis=axis)
  for _ in range(batch_dims):
    take_fn = jax.vmap(take_fn)
  return take_fn(params, indices)


def mask_mean(mask, value, axis=None, drop_mask_channel=False, eps=1e-10):
  """Masked mean."""
  if drop_mask_channel:
    mask = mask[..., 0]

  mask_shape = mask.shape
  value_shape = value.shape

  assert len(mask_shape) == len(value_shape)

  if isinstance(axis, numbers.Integral):
    axis = [axis]
  elif axis is None:
    axis = list(range(len(mask_shape)))
  assert isinstance(axis, collections.Iterable), (
      'axis needs to be either an iterable, integer or "None"')

  broadcast_factor = 1.
  for axis_ in axis:
    value_size = value_shape[axis_]
    mask_size = mask_shape[axis_]
    if mask_size == 1:
      broadcast_factor *= value_size
    else:
      assert mask_size == value_size

  return (jnp.sum(mask * value, axis=axis) /
          (jnp.sum(mask, axis=axis) * broadcast_factor + eps))


def flat_params_to_haiku(params: Mapping[str, np.ndarray]):
  """Convert a dictionary of NumPy arrays to Haiku parameters."""
  hk_params = {}
  for path, array in params.items():
    scope, name = path.split('//')
    if scope not in hk_params:
      hk_params[scope] = {}
    hk_params[scope][name] = jnp.array(array)

  return hk_params


def batched_gather(params, indices, axis=0, batch_dims=0):
  """Implements a JAX equivalent of `tf.gather` with `axis` and `batch_dims`."""
  take_fn = lambda p, i: jnp.take(p, i, axis=axis)
  import jax
  for _ in range(batch_dims):
    take_fn = jax.vmap(take_fn)
  return take_fn(params, indices)


def detached(x:torch.Tensor):
  return x.detach().cpu().numpy()


def unwrap_tensor(x:torch.Tensor):
  return x[0]


def list2tensor(x:list):
  return torch.FloatTensor(np.array(x))


def f2i(x):
  if type(x) is float:
    int_x = int(x)
    if x - int_x <= 0.00000000000001:
      return int_x
    else: # real float
      return x
  else: # other type
    return x


def enum_tree(f, t):
  res_tree = {}
  for k in t.keys():
    if type(t[k]) is dict:
      res_tree[k] = enum_tree(f, t[k])
    else:
      res_tree[k] = f(t[k])
  return res_tree