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

#

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import tensor_util
from tensorflow.python.tools.optimize_for_inference_lib import ensure_graph_is_valid

import numpy as np

def optimize_for_benchmark(input_graph_def, const_dtype, dummy_input):
  ensure_graph_is_valid(input_graph_def)
  optimized_graph_def = change_placehoder_to_const(input_graph_def, const_dtype, dummy_input)
  ensure_graph_is_valid(input_graph_def)
  return optimized_graph_def

def change_placehoder_to_const(input_graph_def, const_dtype, dummy_input):
  result_graph_def = graph_pb2.GraphDef()
  for node in input_graph_def.node:
    if node.op == 'Placeholder':
      new_const = node_def_pb2.NodeDef()
      new_const.op = 'Const'
      new_const.name = node.name
      new_const.attr["dtype"].CopyFrom(node.attr["dtype"])
      tensor_proto = tensor_util.make_tensor_proto(dummy_input,
                                                   const_dtype,
                                                   dummy_input.shape)
      new_const.attr["value"].tensor.CopyFrom(tensor_proto)
      result_graph_def.node.extend([new_const])
    else:
      new_node = node_def_pb2.NodeDef()
      new_node.CopyFrom(node)
      retained_input = []
      for input_node in new_node.input:
        retained_input.append(input_node)
      new_node.input[:] = retained_input

      result_graph_def.node.extend([new_node])

  return result_graph_def
