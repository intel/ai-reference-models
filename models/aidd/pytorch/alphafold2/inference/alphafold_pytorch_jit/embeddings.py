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
from re import I
import torch
from torch import nn
from torch.nn import functional as F
from alphafold_pytorch_jit.basics import dgram_from_positions_pth, NoGatingAttention
from alphafold_pytorch_jit import quat_affine_embed as quat_affine
from alphafold_pytorch_jit import residue_constants
from alphafold_pytorch_jit.backbones import TriangleAttention, TriangleMultiplication, Transition


class TemplatePairSubStack(nn.Module):
  """Pair stack for the templates."""
  def __init__(self, config, global_config, pa_dim):
    super().__init__()
    c = config
    gc = global_config
    self.c = config
    self.global_config = global_config
    self.triangle_attention_starting_node = TriangleAttention(c['triangle_attention_starting_node'], gc,pa_dim)
    self.triangle_attention_ending_node = TriangleAttention(c['triangle_attention_ending_node'], gc,pa_dim)
    self.triangle_multiplication_outgoing = TriangleMultiplication(c['triangle_multiplication_outgoing'], gc,pa_dim)
    self.triangle_multiplication_incoming = TriangleMultiplication(c['triangle_multiplication_incoming'], gc,pa_dim)
    self.pair_transition = Transition(c['pair_transition'], gc, pa_dim)
 
  def forward(self, pair_act, pair_mask):
    # pair_act: [764,764,64]
    # pair_mask: [764,764]
    pair_act = pair_act + self.triangle_attention_starting_node(pair_act,pair_mask)
    pair_act = pair_act + self.triangle_attention_ending_node(pair_act,pair_mask)
    pair_act = pair_act + self.triangle_multiplication_outgoing(pair_act,pair_mask)
    pair_act = pair_act + self.triangle_multiplication_incoming(pair_act,pair_mask)
    pair_act = pair_act + self.pair_transition(pair_act,pair_mask)
    return pair_act


class TemplatePairStack(nn.Module):
  """Pair stack for the templates."""
  def __init__(self, config, global_config,a_dim):
    super().__init__()
    self.c = config
    self.gc = global_config
    self.num_block = self.c['num_block']
    self.template_pair_sub_stack = nn.ModuleList([
      TemplatePairSubStack(self.c,self.gc,a_dim)
      for i in range(self.c['num_block'])
    ])

  def forward(self, pair_act, pair_mask):
    """Builds TemplatePairStack module.

    Arguments:
      pair_act: Pair activations for single template, shape [N_res, N_res, c_t].
      pair_mask: Pair mask, shape [N_res, N_res].
      is_training: Whether the module is in training mode.
      safe_key: Safe key object encapsulating the random number generation key.

    Returns:
      Updated pair_act, shape [N_res, N_res, c_t].
    """
    # pair_act: [764,764,64]
    # pair_mask: [764,764]
    if not self.num_block:
      return pair_act
    for sub_stack in self.template_pair_sub_stack:
      pair_act = sub_stack(pair_act,pair_mask)
    return pair_act


class SingleTemplateEmbedding(nn.Module):

  def __init__(self, config, global_config,a_dim):
    super().__init__()
    self.c = config
    self.dgram_max_bin = self.c['dgram_features']['max_bin']
    self.dgram_min_bin = self.c['dgram_features']['min_bin']
    self.dgram_num_bins = self.c['dgram_features']['num_bins']
    self.use_template_unit_vector = self.c['use_template_unit_vector']
    self.gc = global_config
    n, ca, c = [residue_constants.atom_order[a] for a in ('N', 'CA', 'C')]
    self.res_n = n
    self.res_ca = ca
    self.res_c = c
    num_channels = (self.c['template_pair_stack']
                    ['triangle_attention_ending_node']['value_dim'])
    self.embedding2d = nn.Linear(88,num_channels)
    #  88 is to_concat .shape [-1]
    self.template_pair_stack = TemplatePairStack(self.c['template_pair_stack'],self.gc,a_dim)
    self.output_layer_norm = nn.LayerNorm(normalized_shape=a_dim,elementwise_affine=True)

  def forward(self,
              query_embedding, 
              template_aatype,
              template_pseudo_beta_mask,
              template_pseudo_beta,
              template_all_atom_positions,
              template_all_atom_masks,
              mask_2d):
    # query_embedding: [764, 764, 128]
    # template_aatype: [764]
    # template_pseudo_beta_mask: [764]
    # template_pseudo_beta: [764, 3]
    # template_all_atom_positions: [764, 37, 3]
    # template_all_atom_masks: [764, 37]
    # mask_2d: [764, 764]
    """Build the single template embedding.
    Arguments:
      query_embedding: Query pair representation, shape [N_res, N_res, c_z].
      batch: A batch of template features (note the template dimension has been
        stripped out as this module only runs over a single template).
      mask_2d: Padding mask (Note: this doesn't care if a template exists,
        unlike the template_pseudo_beta_mask).
      is_training: Whether the module is in training mode. 

    Returns:
      A template embedding [N_res, N_res, c_z].
    """
    assert mask_2d.dtype == query_embedding.dtype
    num_res = template_aatype.shape[0]
    template_mask_2d = template_pseudo_beta_mask[:, None] * template_pseudo_beta_mask[None, :]
    #template_mask_2d = template_mask_2d.astype(dtype)
    max_bin = self.dgram_max_bin
    min_bin = self.dgram_min_bin
    num_bins = self.dgram_num_bins
    template_dgram = dgram_from_positions_pth(template_pseudo_beta,
                                          num_bins, min_bin, max_bin)

    to_concat = [template_dgram, template_mask_2d[:, :, None]]
    aatype = F.one_hot(template_aatype.long(), 22)
    to_concat.append(torch.tile(aatype[None, :, :], [num_res, 1, 1]))
    to_concat.append(torch.tile(aatype[:, None, :], [1, num_res, 1]))
    
    # due to reduandant works on transfer quat_affine np2torch
    rot, trans = quat_affine.make_transform_from_reference_pth(
        n_xyz=template_all_atom_positions[:, self.res_n], # 764x3
        ca_xyz=template_all_atom_positions[:, self.res_ca],
        c_xyz=template_all_atom_positions[:, self.res_c])
    affines = quat_affine.QuatAffine_pth(
        quaternion=quat_affine.rot_to_quat_pth(rot, unstack_inputs=True),
        translation=trans,
        rotation=rot,
        unstack_inputs=True)
    ### [Done] function pointer/citation issue
    points = torch.stack([torch.unsqueeze(x, dim=-2) for x in affines.translation])
    affine_vec = affines.invert_point(points, extra_dims=1)
    inv_distance_scalar = 1 / torch.sqrt(
        1e-6 + torch.sum(torch.stack(affine_vec),dim=0))
    ### [Done] sum func with invalid args
    # Backbone affine mask: whether the residue has C, CA, N
    # (the template mask defined above only considers pseudo CB).
    template_mask = (
        template_all_atom_masks[..., self.res_n] *
        template_all_atom_masks[..., self.res_ca] *
        template_all_atom_masks[..., self.res_c])
    template_mask_2d = template_mask[:, None] * template_mask[None, :]
    inv_distance_scalar *= template_mask_2d#.astype(inv_distance_scalar.dtype)
    unit_vector = [(x * inv_distance_scalar)[..., None] for x in affine_vec]
    #unit_vector = [x.astype(dtype) for x in unit_vector]
    #template_mask_2d = template_mask_2d.astype(dtype)
    if not self.use_template_unit_vector:
      ### [Done] Module has no attribute 'c'
      unit_vector = [torch.zeros_like(x) for x in unit_vector]
    to_concat.extend(unit_vector)
    to_concat.append(template_mask_2d[..., None])
    #act = np.concatenate(to_concat, axis=-1)
    act = torch.cat(to_concat, dim=-1)
    # [Done] Arguments for call are not valid.
    # Mask out non-template regions so we don't get arbitrary values in the
    # distogram for these regions.
    act *= template_mask_2d[..., None]
    #act = torch.FloatTensor(act)
    #mask_2d = torch.FloatTensor(mask_2d)
    act = self.embedding2d(act)
    act = self.template_pair_stack(act, mask_2d)
    act = self.output_layer_norm(act)
    return act


class TemplateEmbedding(nn.Module):

  def __init__(self, config, global_config,q_dim):
    super().__init__()
    # q_dim is query_embedding.shape[-1]
    self.c = config
    self.gc = global_config
    self.num_channels = (self.c['template_pair_stack']['triangle_attention_ending_node']['value_dim'])
    self.single_template_embedding = SingleTemplateEmbedding(self.c,self.gc,64)
    # 64 is query_embedding.shape[-1]
    self.attention = NoGatingAttention(self.c['attention'], self.gc,q_dim,self.num_channels,q_dim)
    
  def forward(self, 
              query_embedding, 
              template_mask,
              template_aatype,
              template_pseudo_beta_mask,
              template_pseudo_beta,
              template_all_atom_positions,
              template_all_atom_masks,
              mask_2d):
    # query_embedding: [764, 764, 128]
    # template_mask: [4]
    # template_aatype: [4, 764]
    # template_pseudo_beta_mask: [4, 764]
    # template_pseudo_beta: [4, 764, 3]
    # template_all_atom_positions: [4, 764, 37, 3]
    # template_all_atom_masks: [4, 764, 37]
    # mask_2d: [764, 764]
    """Build TemplateEmbedding module.

    Arguments:
      query_embedding: Query pair representation, shape [N_res, N_res, c_z].
      template_batch: A batch of template features.
      mask_2d: Padding mask (Note: this doesn't care if a template exists,
        unlike the template_pseudo_beta_mask).
      is_training: Whether the module is in training mode.

    Returns:
      A template embedding [N_res, N_res, c_z].
    """
    num_templates = template_mask.shape[0]
    # [done] Module has no attribute 'c'
    num_res = query_embedding.shape[0]
    dtype = query_embedding.dtype
    #template_mask = template_mask
    template_mask = template_mask.to(dtype)
    query_num_channels = query_embedding.shape[-1]
    template_pair_representation = []
    for i in range(template_mask.shape[0]):
      # sampele_batch = {k: template_batch[k][i] for k in template_batch}
      # change sample batch into i-th
      template_pair_representation.append(self.single_template_embedding(
        query_embedding, 
        template_aatype[i],
        template_pseudo_beta_mask[i],
        template_pseudo_beta[i],
        template_all_atom_positions[i],
        template_all_atom_masks[i],
        mask_2d
        ).unsqueeze(0))
    template_pair_representation = torch.cat(template_pair_representation,0)

    # Cross attend from the query to the templates along the residue
    # dimension by flattening everything else into the batch dimension.
    flat_query = torch.reshape(query_embedding,
                             [num_res * num_res, 1, query_num_channels])
    flat_templates = torch.reshape(
        torch.transpose(
          torch.transpose(template_pair_representation, 1, 0),
          1,2
        ),
        [num_res * num_res, num_templates, self.num_channels])
    bias = (1e9 * (template_mask[None, None, None, :] - 1.))
    embedding = self.attention(flat_query, flat_templates,bias)
    embedding = embedding.reshape(num_res,num_res,query_num_channels)
    # No gradients if no templates.
    embedding *= (torch.sum(template_mask) > 0.).to(embedding.dtype)
    # return embedding, template_mask, template_pair_representation, flat_templates
    return embedding

