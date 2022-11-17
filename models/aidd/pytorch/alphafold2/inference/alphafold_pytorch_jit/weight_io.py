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
import os
import torch
import numpy as np
from collections import OrderedDict


hk2pth_name  = {'scale':'weight', 'weights':'weight', 'offset': 'bias'}
iters_name = {'extra_msa_stack','evoformer_iteration','template_pair_sub_stack'}
shape_changed_name = {'weights'}

def load_npy2pth_params(root_path):
  delim = '/'
  assert os.path.isdir(root_path)
  res = {}
  for parent, _, files in os.walk(root_path):
    for f in files:
      if f.endswith('.npy'):
        fp = os.path.join(parent, f)
        data=np.load(fp)
        f_name = f.rstrip('.npy')
        # switch shape
        if f_name in shape_changed_name:
          data = data.swapaxes(-1,-2)
        if f_name in hk2pth_name:
          f_name = hk2pth_name[f_name]
        subparent = parent[len(root_path)+1:]
        jp_flag = 0
        for sub_name in subparent.split(delim):
          # find iterations & change name
          # Before: Iters.Subparent.bias shape(n,x1,x2)
          # After: Iters.1.Subparent.bias shape(x1,x2)
          #        ......
          #        Iters.n.Subparent.bias shape(x1,x2)          
          if sub_name in iters_name:
            jp_flag = 1
            iter_data = torch.from_numpy(data)
            for i in range(data.shape[0]):
              iter_name = subparent.replace(sub_name,sub_name+delim+'{}'.format(i))
              k = os.path.join(iter_name,f_name).replace(delim,'.')
              res[k] = iter_data[i]
        if jp_flag:
          continue
        k = os.path.join(subparent,f_name).replace(delim,'.')
        # if subparent.split('/')
        res[k] = torch.from_numpy(data)
  return res

def load_npy2hk_params(root_path,sample_iter=False):
  """
  get weights from folder npy format
  args: 
      sample_iter: True->return 1 layers params
                   False-> return all layers params
  """
  def sample_leaf_dict_from_stack(root_path):
    """
    return 1 from 1 layer params
    """
    res = {}
    for parent, dirs, files in os.walk(root_path):
      res_sub = {}
      mod_name = os.path.basename(root_path)
      subparent = parent[len(root_path)-len(mod_name):]
      # judge iterable
      iter_flag = 0
      for sub_name in subparent.split('/'):
        if sub_name in iters_name and sub_name != 'template_pair_sub_stack':
          iter_flag = 1
      for f in files:
        if f.endswith('.npy'):
          f_key = f.rstrip('.npy')
          f_path = os.path.join(parent, f)
          if iter_flag == 1:
            # if iterable, for test simple. one layer only
            res_sub[f_key] = np.load(f_path)[0][None]
          else:
            res_sub[f_key] = np.load(f_path)
      res[subparent] = res_sub
    return res

  def get_leaf_dict(root_path,res={}):

    res = {}
    for parent, dirs, files in os.walk(root_path):
      res_sub = {}
      mod_name = os.path.basename(root_path)
      subparent = parent[len(root_path)-len(mod_name):]
      for f in files:
        if f.endswith('.npy'):
          f_key = f.rstrip('.npy')
          f_path = os.path.join(parent, f)
          res_sub[f_key] = np.load(f_path)
      res[subparent] = res_sub
    return res
  assert os.path.isdir(root_path)

  if sample_iter == True :
    res = sample_leaf_dict_from_stack(root_path)
  else:
    res = get_leaf_dict(root_path)
  return res

def filtered_pth_params(pth_params, model:torch.nn.Module):
  res = {}
  for name in list(model.state_dict().keys()):
      res[name] = pth_params[name]
  assert res.keys() == model.state_dict().keys()
  return OrderedDict(res)

if __name__ == '__main__':
  root_path = r'/mnt/data2/alphafold2data/params_extracted/params_model_1/alphafold'
  params_path = os.path.join(root_path,'alphafold_iteration/evoformer/evoformer_iteration/msa_row_attention_with_pair_bias')
  # hk load
  res = load_npy2hk_params(params_path,1)
  print(res)

  # pth load
  state_dict = load_npy2pth_params(params_path)
  print(state_dict.keys())