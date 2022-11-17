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

from typing import Any, Union, Mapping
from alphafold_pytorch_jit import features
import tensorflow.compat.v1 as tf
from torch import nn
import os
import jax
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'alphafold'))
from alphafold.common import confidence
from alphafold_pytorch_jit.subnets import AlphaFold
from alphafold_pytorch_jit.folding import StructureModule
from alphafold_pytorch_jit.utils import detached, unwrap_tensor
from alphafold_pytorch_jit.hk_io import get_pure_fn
from alphafold_pytorch_jit.weight_io import (
  load_npy2hk_params, 
  load_npy2pth_params
)

def get_confidence_metrics(
  prediction_result: Mapping[str, Any]
) -> Mapping[str, Any]:
  """
    post-processes prediction_result to get confidence metrics
  """
  conf_metrics = {}
  conf_metrics['plddt'] = confidence.compute_plddt(
    prediction_result['predicted_lddt']['logits'])
  if 'predicted_aligned_erorr' in prediction_result.keys():
    conf_metrics.update(confidence.compute_predicted_aligned_error(
      prediction_result['predicted_aligned_error']['logits'],
      prediction_result['predicted_aligned_error']['breaks']
    ))
    conf_metrics['ptm'] = confidence.predicted_tm_score(
      prediction_result['predicted_aligned_error']['logits'],
      prediction_result['predicted_aligned_error']['breaks']
    )
  return conf_metrics

def process_features(
    config,
    raw_features: Union[tf.train.Example, features.FeatureDict],
    random_seed: int) -> features.FeatureDict:
  """Processes features to prepare for feeding them into the model.
  Args:
    raw_features: The output of the data pipeline either as a dict of NumPy
      arrays or as a tf.train.Example.
    random_seed: The random seed to use when processing the features.
  Returns:
    A dict of NumPy feature arrays suitable for feeding into the model.
  """
  if isinstance(raw_features, dict):
    return features.np_example_to_features(
        np_example=raw_features,
        config=config,
        random_seed=random_seed)
  else:
    return features.tf_example_to_features(
        tf_example=raw_features,
        config=config,
        random_seed=random_seed)

class RunModel(nn.Module):
  def __init__(self, config, root_params, timer, random_seed) -> None:
    super().__init__()
    ### set hyper params
    mc = config['model']
    gc = mc['global_config']
    sc = mc['heads']['structure_module']
    self.timer = timer
    ### load model params
    self.root_params = root_params
    root_af2iter = os.path.join(root_params, 'alphafold/alphafold_iteration')
    root_struct = os.path.join(root_af2iter, 'structure_module')
    af2iter_params = load_npy2pth_params(root_af2iter)
    struct_params = load_npy2hk_params(root_struct)
    struct_rng = jax.random.PRNGKey(random_seed)
    ### create compatible structure module
    # time cost is low at structure-module
    # no need to cvt it to PyTorch version
    _, struct_apply = get_pure_fn(StructureModule, sc, gc)
    ### create AlphaFold instance
    #evo_init_dims = {
    #  'target_feat':batch['target_feat'].shape[-1],
    #  'msa_feat':batch['msa_feat'].shape[-1]
    #}
    evo_init_dims = {
      'target_feat': 22,
      'msa_feat': 49
    }
    self.model = AlphaFold(
      mc,
      evo_init_dims,
      af2iter_params,
      struct_apply,
      struct_params,
      struct_rng,
      'alphafold',
      timer=self.timer
    )
  
  def forward(self, feat):
    timer_name = 'model_inference'
    self.timer.add_timmer(timer_name)
    # [inc] unwrap batch data if data is unsuqeeze by INC
    if feat['seq_length'].dim() > 1:
      print('### [INFO] INC input detected')
      feat = jax.tree_map(unwrap_tensor, feat)
    result = self.model(feat)
    #del feat
    #result = jax.tree_map(cvt_result, result)
    result = jax.tree_map(detached, result)
    result.update(get_confidence_metrics(result))
    self.timer.end_timmer(timer_name)
    self.timer.save()
    return result
