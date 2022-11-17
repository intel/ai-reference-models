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
import numpy as np
from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F
from alphafold_pytorch_jit.folding import all_atom
from alphafold_pytorch_jit.basics import pseudo_beta_fn_no_masks, dgram_from_positions_pth, create_extra_msa_feature
from alphafold_pytorch_jit.backbones import ExtraEvoformerIteration, NoExtraEvoformerIteration
from alphafold_pytorch_jit.embeddings import TemplateEmbedding
from alphafold_pytorch_jit.heads import (
  ExperimentallyResolvedHead, 
  PredictedAlignedErrorHead, 
  PredictedLDDTHead, 
  MaskedMsaHead,
  DistogramHead)
from alphafold_pytorch_jit import residue_constants
from alphafold_pytorch_jit.weight_io import filtered_pth_params
from alphafold_pytorch_jit.utils import detached, list2tensor
import jax
import time
#from tqdm import tqdm
#import pdb


class EmbeddingsAndEvoformer(nn.Module):

  def __init__(self,config, global_config,init_dims):
    super().__init__()
    self.c = config
    self.gc = global_config
    self.recycle_pos = self.c['recycle_pos']
    self.num_bins = self.c['prev_pos']['num_bins']
    self.min_bins = self.c['prev_pos']['min_bin']
    self.max_bins = self.c['prev_pos']['max_bin']
    self.recycle_features = self.c['recycle_features']
    self.max_relative_feature = self.c['max_relative_feature']
    self.template_enabled = self.c['template']['enabled']
    self.template_embed_torsion_angles = self.c['template']['embed_torsion_angles']
    self.zero_init = self.gc['zero_init']
    # init_dims = {'target_feat':22, 'msa_feat':49}
    self.preprocess_1d = nn.Linear(init_dims['target_feat'], self.c['msa_channel'])
    self.preprocess_msa = nn.Linear(init_dims['msa_feat'], self.c['msa_channel'])
    self.left_single = nn.Linear(init_dims['target_feat'], self.c['pair_channel'])
    self.right_single = nn.Linear(init_dims['target_feat'], self.c['pair_channel'])
    self.extra_msa_activations = nn.Linear(25,self.c['extra_msa_channel'])
    print('### [INFO] build evoformer network')
    self.extra_msa_stack = nn.ModuleList([
                          ExtraEvoformerIteration(
                            self.c['evoformer'], 
                            self.gc,
                            True, # is_extra_msa is True
                            self.c['extra_msa_channel'], 
                            self.c['extra_msa_channel'], 
                            self.c['pair_channel'])
                          for i in range(self.c['extra_msa_stack_num_block'])])
    self.evoformer_iteration = nn.ModuleList([
                          NoExtraEvoformerIteration(
                            self.c['evoformer'], 
                            self.gc,
                            False, # is_extra_msa is False
                            self.c['msa_channel'], 
                            self.c['msa_channel'], 
                            self.c['pair_channel'])
                          for i in range(self.c['evoformer_num_block'])])
    self.single_activations = nn.Linear(self.c['msa_channel'],self.c['seq_channel'])
    self.prev_pos_linear = nn.Linear(15,self.c['pair_channel'])
    self.prev_msa_first_row_norm = nn.LayerNorm(normalized_shape=256,elementwise_affine=True)
    self.prev_pair_norm = nn.LayerNorm(normalized_shape=128,elementwise_affine=True)
    self.pair_activiations = nn.Linear(65,self.c['pair_channel'])
    self.template_embedding = TemplateEmbedding(
      self.c['template'],self.gc,self.c['pair_channel'])
    self.template_single_embedding = nn.Linear(57,self.c['msa_channel'])
    self.template_projection = nn.Linear(self.c['msa_channel'],self.c['msa_channel'])
    
  @torch.jit.ignore
  def read_time(self) -> float:
    return time.time()
  
  def forward(self,
              target_feat,
              msa_feat,
              seq_mask,
              aatype,
              prev_pos,
              prev_msa_first_row,
              prev_pair,
              residue_index,
              template_mask,
              template_aatype,
              template_pseudo_beta_mask,
              template_pseudo_beta,
              template_all_atom_positions,
              template_all_atom_masks,
              extra_msa,
              extra_has_deletion,
              extra_deletion_value,
              extra_msa_mask,
              msa_mask,
              torsion_angles_sin_cos,
              alt_torsion_angles_sin_cos,
              torsion_angles_mask
    ):
    ### start here: computing stuck at 2nd alphafold_iteration
    t1_embedding = self.read_time()
    print('  # [INFO] linear embedding of features')
    preprocess_1d = self.preprocess_1d(target_feat)
    preprocess_msa = self.preprocess_msa(msa_feat)
    msa_activations = torch.unsqueeze(preprocess_1d, dim=0) + preprocess_msa
    print('  # [INFO] embedding left/right single ')
    left_single = self.left_single(target_feat)
    right_single = self.right_single(target_feat)
    pair_activations = left_single[:, None] + right_single[None]
    mask_2d = seq_mask[:, None] * seq_mask[None, :]
    if self.recycle_pos and prev_pos.shape != 0:
      print('  # [INFO] embedding previous molecular graph ')
      prev_pseudo_beta = pseudo_beta_fn_no_masks(aatype, prev_pos)
      dgram = dgram_from_positions_pth(prev_pseudo_beta, self.num_bins, self.min_bins, self.max_bins)
      pair_activations += self.prev_pos_linear(dgram)
    if self.recycle_features:
      print('  # [INFO] recycle previous molecular graph ')
      if prev_msa_first_row.shape != 0:
        prev_msa_first_row = self.prev_msa_first_row_norm(prev_msa_first_row)
        msa_activations[0] += prev_msa_first_row
      if prev_pair.shape != 0:
        pair_activations += self.prev_pair_norm(prev_pair)
    if self.max_relative_feature:
      print('  # [INFO] cvt residue features to one-hot format ')
      pos = residue_index
      offset = pos[:, None] - pos[None, :]
      rel_sub_input = offset.long() + self.max_relative_feature
      rel_pos = F.one_hot(
          rel_sub_input.clip( 
              min=0,
              max=int(2 * self.max_relative_feature)
              ).long(),
          2 * self.max_relative_feature + 1)
      pair_activations += self.pair_activiations(rel_pos.float())
    ### stop here: computing stuck at 2nd alphafold_iteration

    if self.template_enabled:
      print(' ## [INFO] execute template embedding')
      template_pair_representation = self.template_embedding(
                                      pair_activations,
                                      template_mask,
                                      template_aatype,
                                      template_pseudo_beta_mask,
                                      template_pseudo_beta,
                                      template_all_atom_positions,
                                      template_all_atom_masks,
                                      mask_2d)
      pair_activations += template_pair_representation
    # Embed extra MSA features.
    print(' ## [INFO] execute extra_msa_activations')
    extra_msa_feat = create_extra_msa_feature(extra_msa,extra_has_deletion,extra_deletion_value)
    extra_msa_activations = self.extra_msa_activations(extra_msa_feat)
    # Extra MSA Stack.
    print(' ## [INFO] execute extra_msa_iterations')
    n_msa_iters = len(self.extra_msa_stack)
    for i, extra_msa_iter in enumerate(self.extra_msa_stack):
      print('  # [INFO] execute extra_msa_iter %d/%d' % (i+1, n_msa_iters))
      extra_msa_output = extra_msa_iter(
        extra_msa_activations,
        pair_activations,
        extra_msa_mask,
        mask_2d)
      ### update input for next step
      extra_msa_activations = extra_msa_output['msa']
      pair_activations = extra_msa_output['pair']
    
    # prepare input for evoformer stack
    evoformer_input = {}
    evoformer_input['msa'] = msa_activations
    evoformer_input['pair'] = pair_activations
    evoformer_masks = {}
    evoformer_masks['msa'] = msa_mask
    evoformer_masks['pair'] = mask_2d
    # Append num_templ rows to msa_activations with template embeddings.
    if self.template_enabled and self.template_embed_torsion_angles:
      print(' ## [INFO] execute template projection')
      num_templ, num_res = template_aatype.shape
      aatype_one_hot = F.one_hot(template_aatype.long(),22)
      ret = {}
      ret['torsion_angles_sin_cos'] = torsion_angles_sin_cos
      ret['alt_torsion_angles_sin_cos'] = alt_torsion_angles_sin_cos
      ret['torsion_angles_mask'] = torsion_angles_mask
      template_features = torch.cat([
        aatype_one_hot.to(torch.float32),
        torch.reshape(ret['torsion_angles_sin_cos'], [num_templ, num_res, 14]),
        torch.reshape(ret['alt_torsion_angles_sin_cos'], [num_templ, num_res, 14]),
        ret['torsion_angles_mask']]
      , dim=-1)
      template_activations = self.template_single_embedding(template_features)
      template_activations = F.relu(template_activations)
      template_activations = self.template_projection(template_activations)
      # Concatenate the templates to the msa.
      evoformer_input['msa'] = torch.cat(
          [evoformer_input['msa'],
          template_activations],
          dim=0)
      # Concatenate templates masks to the msa masks.
      # Use mask from the psi angle, as it only depends on the backbone atoms
      # from a single residue.
      torsion_angle_mask = ret['torsion_angles_mask'][:, :, 2]
      # Concatenate the templates to the msa.
      evoformer_masks['msa'] = torch.cat(
          [evoformer_masks['msa'],
          torsion_angle_mask],
          dim=0)
    t2_embedding = self.read_time()
    print('  # [TIME] total embedding duration =', (t2_embedding - t1_embedding), 'sec')

    print(' ## [INFO] execute evoformer_iterations')
    t1_evoformer=  self.read_time()
    for i, evoformer_iter in enumerate(self.evoformer_iteration):
      evoformer_input = evoformer_iter(
        evoformer_input['msa'],
        evoformer_input['pair'],
        evoformer_masks['msa'],
        evoformer_masks['pair']
      )
    # evoformer_input is output of iterations
    evoformer_output = evoformer_input
    msa_activations = evoformer_output['msa']
    pair_activations = evoformer_output['pair']
    single_activations = self.single_activations(msa_activations[0])
    num_sequences = msa_feat.shape[0]
    output = {}
    output['single'] = single_activations
    output['pair'] = pair_activations
    output['msa'] = msa_activations[:num_sequences, :, :]
    output['msa_first_row'] = msa_activations[0]
    t2_evoformer = self.read_time()
    print('  # [TIME] total evoformer duration =', (t2_evoformer - t1_evoformer), 'sec')
    return output


class AlphaFold(nn.Module):
  def __init__(self, 
    config, 
    evo_init_dims,
    af2iter_params,
    struct_apply,
    struct_params,
    struct_rng,
    name='alphafold',
    timer=None
  ):
    super().__init__()
    self.c = config
    self.config = config
    self.gc = config['global_config']
    self.timer = timer
    self.recycle_iter_idx = -1
    ### modules
    self.impl = AlphaFoldIteration(
      self.config, 
      self.gc, 
      evo_init_dims,
      struct_apply
    )
    ### filter input params
    af2iter_params = filtered_pth_params(af2iter_params, self.impl)
    self.impl.load_state_dict(af2iter_params)
    self.struct_params = struct_params
    self.struct_rng = struct_rng

  ### save current representations to "prev_" keys
  # s.t. current keys can be reused by next recycle
  def _get_prev(self, ret):
    new_prev = {}
    new_prev['prev_pos'] = ret['structure_module']['final_atom_positions']
    new_prev['prev_msa_first_row'] = ret['representations']['msa_first_row']
    new_prev['prev_pair'] = ret['representations']['pair']
    return new_prev
  
  ### forward func for current recycle
  def _do_call(self,
    prev,
    recycle_idx,
    batch,
    ensemble_representations
  ):
    batch_size = batch['aatype'].shape[0]
    ### resample MSA data from indicated slice if defined
    if self.config['resample_msa_in_recycling']:
      num_ensemble = batch_size // (self.config['num_recycle'] + 1)
      ### helper func: slice out the batch of current recycle
      def slice_recycle_idx(x):
        x = x.detach().cpu().numpy()
        start = recycle_idx * num_ensemble
        size = num_ensemble
        res = torch.tensor(np.array(
          jax.lax.dynamic_slice_in_dim(x, start, size, axis=0)))
        return res
      ensembled_batch = jax.tree_map(slice_recycle_idx, batch)
      ensembled_batch = jax.tree_map(torch.tensor, ensembled_batch)
      del batch # 
    else:
      num_ensemble = batch_size
      ensembled_batch = batch
    ### update recycling counter
    self.recycle_iter_idx += 1
    ### launch AF2 iteration for current recycle
    return self.impl(
      ensembled_batch = ensembled_batch,
      Struct_Params = self.struct_params,
      rng=self.struct_rng,
      non_ensembled_batch = prev,
      ensemble_representations=ensemble_representations,
      idx=recycle_idx
    )

  def forward(
    self, 
    batch,
    compute_loss=False,
    ensemble_representations=False,
    return_representations=False
  ):
    print('### [INFO] jit compilation') # [issue] PyTorch 1.11 has a bug at 2nd alphafold iter
    self.impl.compile() # use jit.script to compile alphafolditeration
    num_residues = batch['aatype'].shape[1]
    ### reset "prev_" keys for current recycling if defined
    if self.config['num_recycle']:
      emb_config = self.config['embeddings_and_evoformer']
      prev = {
        'prev_pos' : torch.zeros(
          (num_residues, residue_constants.atom_type_num, 3)),
        'prev_msa_first_row' : torch.zeros(
          (num_residues, emb_config['msa_channel'])),
        'prev_pair' : torch.zeros(
          (num_residues, num_residues, emb_config['pair_channel']))
      }
      ### only reset num_iter to min if train
      if 'num_iter_recycling' in batch:
        num_iter = batch['num_iter_recycling'][0]
        num_iter = np.min(num_iter, self.config['num_recycle'])
      else: # directly set max number of iter if test
        num_iter = self.config['num_recycle']
      ### recycling loop
      #num_iter = 0 # [inc TODO] debug for INC, please remove this flag after debug finished
      for i in range(0, num_iter+1):
        print('### [INFO] start AlphaFold Iteration-%d' % (i+1))
        t0 = time.time()
        res = self._do_call(
          prev,
          i,
          batch,
          ensemble_representations
        )
        if i < num_iter:
          print('  # [INFO] save curr update as previous output.')
          prev = self._get_prev(res)
          print('  # [INFO] update to prev done.')
        dt = time.time() - t0
        print('  # [INFO] duration = %.2fs' % dt)
    else: # 1 iteration if num_iter is not defined
      res = self._do_call(
        {},
        1,
        batch,
        ensemble_representations
      )
    if compute_loss:
      res = res[0], [res[1]]
    if not return_representations:
      del (res[0] if compute_loss else res)['representations']
    del batch
    return res


class AlphaFoldIteration(nn.Module):

  def __init__(self, config, global_config, evo_init_dims,struct_apply,name='alphafold_iteration'):
    super().__init__()
    self.c = config
    self.gc = global_config
    self.evoformer = EmbeddingsAndEvoformer(
      self.c['embeddings_and_evoformer'], 
      self.gc,
      evo_init_dims)
    self.heads = OrderedDict()
    for head_name, head_config in sorted(self.c['heads'].items()):
      if not head_config['weight'] or head_name in ['structure_module']:
        continue  # Do not instantiate zero-weight heads.
      head_factory = {
          'masked_msa': MaskedMsaHead,
          'distogram': DistogramHead,
          'predicted_lddt': PredictedLDDTHead,
          'predicted_aligned_error': PredictedAlignedErrorHead,
          'experimentally_resolved': ExperimentallyResolvedHead,
      }[head_name]
      self.heads[head_name] = (
                          head_factory(head_config, self.gc))
    self.heads['structure_module'] = struct_apply

  def _slice_batch(self, i, ensembled_batch, non_ensembled_batch):
    b = {k: v[i] for k, v in ensembled_batch.items()}
    if non_ensembled_batch is not None: # omit if prev-keys not exist
      b.update(non_ensembled_batch) # fuse previous representation
    return b

  def _body(self, x, ensembled_batch, non_ensembled_batch, idx):
    """Add one element to the representations ensemble."""
    i, current_representations = x
    del x
    feats = self._slice_batch(i, ensembled_batch, non_ensembled_batch)
    representations_update = self.evoformer(*self.batch_expand(feats))
    new_representations = {}
    for k in current_representations:
      new_representations[k] = (
          current_representations[k] + representations_update[k])
    del representations_update
    return i+1, new_representations

  def batch_expand(self, batch:dict):
    """
    expand batch into long list for jit accleration
    """
    values_keys_order=[
      'target_feat',
      'msa_feat',
      'seq_mask',
      'aatype',
      'prev_pos',
      'prev_msa_first_row',
      'prev_pair',
      'residue_index',
      'template_mask',
      'template_aatype',
      'template_pseudo_beta_mask',
      'template_pseudo_beta',
      'template_all_atom_positions',
      'template_all_atom_masks',
      'extra_msa',
      'extra_has_deletion',
      'extra_deletion_value',
      'extra_msa_mask',
      'msa_mask'] 
    ordered_values=[]
    for i in values_keys_order:
      ordered_values.append(batch[i])
    print('  [INFO] atom37 -> torsion angles')
    ret = all_atom.atom37_to_torsion_angles(
      aatype=np.array(batch['template_aatype']),
      all_atom_pos=np.array(batch['template_all_atom_positions']),
      all_atom_mask=np.array(batch['template_all_atom_masks']),
      # Ensure consistent behaviour during testing:
      placeholder_for_undefined=not self.gc['zero_init'])
    for i in ret.values():
      ordered_values.append(torch.FloatTensor(i))
    return ordered_values

  def compile(self):
    self.evoformer = torch.jit.optimize_for_inference(torch.jit.script(self.evoformer))
    # torch.jit.trace
  
  @torch.jit.ignore
  def read_time(self) -> float:
    return time.time()
 
  def forward(self,
               ensembled_batch,
               Struct_Params,
               rng,
               non_ensembled_batch=None,
               ensemble_representations=True,
               idx=-1
  ):
    num_ensemble = ensembled_batch['seq_length'].shape[0]
    if not ensemble_representations:
      assert ensembled_batch['seq_length'].shape[0] == 1
    '''EmbeddingsAndEvoformer part'''
    batch0 = self._slice_batch(0, ensembled_batch, non_ensembled_batch)
    evo_input=self.batch_expand(batch0)
    print('  # [INFO] start evoformer iteration',idx)
    representations = self.evoformer(*evo_input)
    #print(self.evoformer.graph)
    msa_representation = representations['msa']
    if ensemble_representations:
      x = [1,representations]
      while x[0] < num_ensemble:
        x = self._body(x, ensembled_batch, non_ensembled_batch, idx)
      representations = x[1]
      for k in representations:
        if k != 'msa':
          representations[k] /= num_ensemble
    representations['msa'] = msa_representation

    '''structure_module part'''
    ret = {}
    ret['representations'] = representations
    t1_head = self.read_time()
    for name, (module) in self.heads.items():
      if name in ('predicted_lddt', 'predicted_aligned_error'):
        continue
      elif name in ['structure_module']:
        representations_hk = jax.tree_map(detached,representations)
        batch_hk = jax.tree_map(detached,batch0)
        res_hk = module(Struct_Params,rng,representations_hk,batch_hk)
        ret[name] = jax.tree_map(list2tensor,res_hk)
        del res_hk
        if 'representations' in ret[name]:
          representations.update(ret[name].pop('representations'))
          # print('# ====> [INFO] pLDDTHead input has been saved.')
          # with open('structure_module_input.pkl', 'wb') as h_tmp:
          #   pickle.dump(representations['structure_module'], h_tmp, protocol=4)
      else:
        ret[name] = module(representations)
        if 'representations' in ret[name]:
          representations.update(ret[name].pop('representations'))
      
    if self.c['heads']['predicted_lddt']['weight']:
      # Add PredictedLDDTHead after StructureModule executes.
      name = 'predicted_lddt'
      # Feed all previous results to give access to structure_module result.
      module = self.heads[name]
      ret[name] = module(representations)
    if ('predicted_aligned_error' in self.c['heads']
        and self.c['heads']['predicted_aligned_error']['weight']):
      # Add PredictedAlignedErrorHead after StructureModule executes.
      name = 'predicted_aligned_error'
      # Feed all previous results to give access to structure_module result.
      module = self.heads[name]
      ret[name] = module(representations)
    t2_head = self.read_time()
    print('  # [TIME] total heads duration =', (t2_head - t1_head), 'sec')
    #del representations
    return ret

