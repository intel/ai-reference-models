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
from typing import Mapping
import numpy as np
import os

FeatureDict = Mapping[str, np.ndarray]

def save_feature_dict(f:str, data:FeatureDict):
  np.savez(f, **data)

def load_feature_dict(f:str) -> FeatureDict:
  df = np.load(f, allow_pickle=True)
  res = {}
  for k in df.files:
    res[k] = df[k]
  return res

def load_feature_dict_if_exist(f:str):
  if os.path.exists(f):
    return load_feature_dict(f)
  else:
    return None

def get_mock_2darray(h, w):
  np.random.seed(1)
  if h == 0:
    return np.array(['type1', 'type2', 'type3'])
  else:
    return np.random.random((h, w)) if h % 4 == 0 else (np.random.random((h, w)) * 255).astype(np.int)

if __name__ == '__main__':
  f = 'features.pkl'
  data = None
  import pickle
  with open(f, 'rb') as h:
    data = pickle.load(h)
  
  import pdb
  pdb.set_trace()