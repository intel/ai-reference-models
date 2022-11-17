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
import numpy as np
import argparse
from tqdm import tqdm
import shutil

def extract_param_npz(f, root_output):
  if not os.path.isdir(root_output):
    os.mkdir(root_output)
  df = np.load(f)
  keys = list(df.keys())
  delim= '//'
  if len(root_output) < 1:
    root_output = f.rstrip('.npz')
  for k in tqdm(keys):
    d, prefix = k.split(delim)
    subd = os.path.join(root_output, d)
    if not os.path.isdir(subd):
      os.makedirs(subd)
    fp_out = os.path.join(subd, '%s.npy' % prefix)
    np.save(fp_out, df[k])

def fix_name_issue(root_params):
  ### weights with prefix __ needs to rename before use
  # root_params: 
  #   alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack/
  # src subdir: __layer_stack_no_state
  # dst subdir: template_pair_sub_stack
  assert root_params.rstrip('/\\').endswith('template_pair_stack')
  src_subdir = os.path.join(root_params, '__layer_stack_no_state')
  assert os.path.isdir(src_subdir)
  dst_subdir = os.path.join(root_params, 'template_pair_sub_stack')
  if not os.path.isdir(dst_subdir):
    print('[params migration]\n  %s\n  ->\n  %s' % (src_subdir, dst_subdir))
    shutil.move(src_subdir, dst_subdir)


parser = argparse.ArgumentParser('Load and extract indicated alphafold2 model parameter.npz file')
parser.add_argument('--input', type=str, default='/mnt/data1/params/params_model_1.npz')
parser.add_argument('--output_dir', type=str, default='/mnt/data1/af2home/weights/extracted/model_1')
args = parser.parse_args()
if not os.path.isdir(args.output_dir):
  extract_param_npz(args.input, args.output_dir)
root_issue = os.path.join(
  args.output_dir,
  'alphafold/alphafold_iteration/evoformer/template_embedding/single_template_embedding/template_pair_stack')
fix_name_issue(root_issue)

