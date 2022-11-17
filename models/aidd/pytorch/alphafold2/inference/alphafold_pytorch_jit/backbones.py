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
import pdb
import torch
from torch import nn
from alphafold_pytorch_jit.basics import GatingAttention, MSAColumnAttention, MSARowAttentionWithPairBias, MSAColumnGlobalAttention


class Transition(nn.Module):

  def __init__(self,config, global_config, act_dim):
    super().__init__()
    self.config = config
    self.global_config = global_config
    num_intermediate = int(act_dim * self.config['num_intermediate_factor'])
    self.input_layer_norm = nn.LayerNorm(normalized_shape=act_dim,elementwise_affine=True)
    self.transition1 = nn.Linear(act_dim,num_intermediate)
    self.relu = nn.ReLU()
    self.transition2 = nn.Linear(num_intermediate,act_dim)


  def forward(self, act, mask):
    """ Transition adaptor of MSA representation & template matching
    Arguments:
      act: Query tensor batch_size x N_res x N_channel.
      mask: mask tensor batch_size x N_res.
    Returns:
      FP32 Tensor batch_size x N_res x N_channel.
    """
    act = self.input_layer_norm(act)
    act = self.transition1(act)
    act = self.relu(act)
    act = self.transition2(act)
    return act


class OuterProductMean(nn.Module):

  def __init__(self,config, global_config,act_dim,num_output_channel):
    super().__init__()    
    """Builds OuterProductMean module.

    Arguments:
      act: MSA representation, shape [N_seq, N_res, c_m].
      mask: MSA mask, shape [N_seq, N_res].
      is_training: Whether the module is in training mode.

    Returns:
      Update to pair representation, shape [N_res, N_res, c_z].
    """
    self.config = config
    self.global_config = global_config
    c = config
    gc = global_config
    self.num_output_channel = num_output_channel
    self.layer_norm_input = nn.LayerNorm(normalized_shape=act_dim,elementwise_affine=True)
    self.left_projection = nn.Linear(act_dim,self.config['num_outer_channel'])
    self.right_projection = nn.Linear(act_dim,self.config['num_outer_channel'])
    self.output_w = nn.Parameter(torch.Tensor(
                              c['num_outer_channel'], c['num_outer_channel'],self.num_output_channel))
    self.output_b = nn.Parameter(torch.Tensor(self.num_output_channel,))

  def compute_chunk(self,left_act,right_act):
    left_act = torch.transpose(left_act, 2, 1)
    act = torch.einsum('acb,ade->dceb', left_act, right_act)
    act = torch.einsum('dceb,cef->dbf', act, self.output_w) + self.output_b
    return torch.transpose(act, 1, 0)

  def forward(self, act, mask):
    mask = mask[..., None]
    act = self.layer_norm_input(act)
    left_act = mask * self.left_projection(act)
    right_act = mask * self.right_projection(act)
    act = self.compute_chunk(left_act,right_act)
    epsilon = 1e-3
    norm = torch.einsum('abc,adc->bdc', mask, mask)
    act /= epsilon + norm
    return act


class TriangleMultiplication(nn.Module):

  def __init__(self,config, global_config, act_dim):
    """Builds TriangleMultiplication module.

    Arguments:
      act: Pair activations, shape [N_res, N_res, c_z]
      mask: Pair mask, shape [N_res, N_res].
      is_training: Whether the module is in training mode.

    Returns:
      Outputs, same shape/type as act.
    """
    super().__init__()
    self.config = config
    self.global_config = global_config
    self.c_equation = self.config['equation']
    self.layer_norm_input = nn.LayerNorm(normalized_shape=act_dim,elementwise_affine=True)
    self.left_projection = nn.Linear(act_dim,self.config['num_intermediate_channel'])
    self.right_projection = nn.Linear(act_dim,self.config['num_intermediate_channel'])
    self.left_gate = nn.Linear(act_dim,self.config['num_intermediate_channel'])
    self.right_gate = nn.Linear(act_dim,self.config['num_intermediate_channel'])
    self.center_layer_norm = nn.LayerNorm(normalized_shape=act_dim,elementwise_affine=True)
    self.output_projection = nn.Linear(act_dim,act_dim)
    self.gating_linear = nn.Linear(act_dim,act_dim)

  def forward(self, act, mask):
    mask = mask[..., None]
    act = self.layer_norm_input(act)
    input_act = act # For gate
    left_proj_act = mask * self.left_projection(act)
    right_proj_act = mask * self.right_projection(act)
    left_proj_act *= torch.sigmoid(self.left_gate(act))
    right_proj_act *= torch.sigmoid(self.right_gate(act))
    # "Outgoing" edges equation: 'ikc,jkc->ijc'
    # "Incoming" edges equation: 'kjc,kic->ijc'
    # Note on the Suppl. Alg. 11 & 12 notation:
    # For the "outgoing" edges, a = left_proj_act and b = right_proj_act
    # For the "incoming" edges, it's swapped:
    #   b = left_proj_act and a = right_proj_act
    act = torch.einsum(self.c_equation, left_proj_act, right_proj_act)
    act = self.center_layer_norm(act)
    act = self.output_projection(act)
    act *= torch.sigmoid(self.gating_linear(input_act))
    return act


class TriangleAttention(nn.Module):

  def __init__(self,config, global_config, pa_dim):
    """Builds TriangleAttention module.

    Arguments:
      pair_act: [N_res, N_res, c_z] pair activations tensor
      pair_mask: [N_res, N_res] mask of non-padded regions in the tensor.
      is_training: Whether the module is in training mode.

    Returns:
      Update to pair_act, shape [N_res, N_res, c_z].
    """
    super().__init__()
    self.c = config
    self.gc = global_config
    self.c_orientation = self.c['orientation']
    assert self.c['orientation'] in ['per_row', 'per_column']
    self.query_norm = nn.LayerNorm(normalized_shape=pa_dim,elementwise_affine=True)
    self.feat_2d_weights = nn.Parameter(torch.Tensor(pa_dim,self.c['num_head']))
    # self.c['gating'] is 1, use GatingAttention
    self.attention = GatingAttention(self.c,self.gc,pa_dim,pa_dim,pa_dim)

  def _slice_attention(self,q_data,m_data,bias,nonbatched_bias=torch.Tensor()):
    """get same result with sliced input."""
    ### avoiding huge memory cost
    ### threhold is ajustable
    threhold = 1000
    unit = 320 # unit is adjustable
    if q_data.size()[0] > threhold:
      res = torch.ones_like(q_data)
      for i in range(q_data.size()[0] // unit):
        q_sub_data = q_data[unit*i:unit*(i+1)]
        m_sub_data = m_data[unit*i:unit*(i+1)]
        bias_sub = bias[unit*i:unit*(i+1)]
        res[unit*i:unit*(i+1)] = self.attention(q_sub_data,m_sub_data,bias_sub,nonbatched_bias)
      return res
    else:
      return self.attention(q_data,m_data,bias,nonbatched_bias)

  def forward(self, pair_act, pair_mask):
    assert pair_act.dim() == 3
    assert pair_mask.dim() == 2
    if self.c_orientation == 'per_column':
      pair_act = torch.swapaxes(pair_act, -2, -3)
      pair_mask = torch.swapaxes(pair_mask, -1, -2)
    bias = (1e9 * (pair_mask - 1.))[:, None, None, :]
    assert bias.dim() == 4
    pair_act = self.query_norm(pair_act)
    nonbatched_bias = torch.einsum('qkc,ch->hqk', pair_act, self.feat_2d_weights)
    pair_act = self._slice_attention(pair_act,pair_act,bias,nonbatched_bias)
    if self.c_orientation == 'per_column':
      pair_act = torch.swapaxes(pair_act, -2, -3)
    return pair_act    



### [done] below: need rm logic branchs in __init__
# EvoformerIteration => ExtraEvoformerIteration & NoExtraEvoformerIteration


class NoExtraEvoformerIteration(nn.Module):
  
  def __init__(self, config, global_config, is_extra_msa,a_dim, m_dim, pa_dim):
    super().__init__()
    """Builds EvoformerIteration module.

    Arguments:
      activations: Dictionary containing activations:
        * 'msa': MSA activations, shape [N_seq, N_res, c_m].
        * 'pair': pair activations, shape [N_res, N_res, c_z].
      masks: Dictionary of masks:
        * 'msa': MSA mask, shape [N_seq, N_res].
        * 'pair': pair mask, shape [N_res, N_res].
      is_training: Whether the module is in training mode.
      safe_key: prng.SafeKey encapsulating rng key.

    Returns:
      Outputs, same shape/type as act.
    """
    self.config = config
    self.global_config = global_config
    c = config
    gc = global_config
    self.is_extra_msa = is_extra_msa
    # a_dim = m_dim = 64, pa_dim = 128
    self.msa_row_attention_with_pair_bias = MSARowAttentionWithPairBias(c['msa_row_attention_with_pair_bias'],gc,a_dim, m_dim, pa_dim)
    ### this is real annoucement
    self.msa_column_attention = MSAColumnAttention(c['msa_column_attention'],gc,a_dim,a_dim,a_dim)
      #del self.msa_column_global_attention
    self.msa_transition = Transition(c['msa_transition'], gc,a_dim) # a_dim=64
    # a_dim = 64, pa_dim = 128
    self.outer_product_mean = OuterProductMean(c['outer_product_mean'], gc,a_dim,pa_dim)
    self.triangle_multiplication_outgoing = TriangleMultiplication(c['triangle_multiplication_outgoing'], gc,pa_dim)
    self.triangle_multiplication_incoming = TriangleMultiplication(c['triangle_multiplication_incoming'], gc,pa_dim)
    self.triangle_attention_starting_node = TriangleAttention(c['triangle_attention_starting_node'], gc,pa_dim)
    self.triangle_attention_ending_node = TriangleAttention(c['triangle_attention_ending_node'], gc,pa_dim)
    self.pair_transition = Transition(c['pair_transition'], gc, pa_dim) # pa_dim=128


  def forward(self, msa_act, pair_act, msa_mask, pair_mask):
    #msa_act, pair_act = activations['msa'], activations['pair']
    #msa_mask, pair_mask = masks['msa'], masks['pair']
    msa_act = msa_act + self.msa_row_attention_with_pair_bias(msa_act, msa_mask, pair_act=pair_act)
    msa_act = msa_act + self.msa_column_attention(msa_act,msa_mask)
    # msa_transition_input.msa_act: torch.Tensor [5120, 206, 64]
    # msa_transition_input.msa_mask: torch.Tensor [5120, 206]
    msa_act = msa_act + self.msa_transition(msa_act,msa_mask)
    # msa_act [5120, 206, 64]
    # msa_mask [5120, 206]
    # pair_act [206, 206, 128]
    pair_act = pair_act + self.outer_product_mean(msa_act,msa_mask)
    res = {'msa':msa_act}
    pair_act = pair_act + self.triangle_multiplication_outgoing(pair_act,pair_mask)
    pair_act = pair_act + self.triangle_multiplication_incoming(pair_act,pair_mask)
    pair_act = pair_act + self.triangle_attention_starting_node(pair_act,pair_mask)
    pair_act = pair_act + self.triangle_attention_ending_node(pair_act,pair_mask)
    # pair_trainsition_input.pair_act: torch.Tensor [206, 206, 128]
    # pair_trainsition_input.pair_mask: torch.Tensor [206, 206]
    pair_act = pair_act + self.pair_transition(pair_act,pair_mask)
    res['pair'] = pair_act
    return res


class ExtraEvoformerIteration(nn.Module):
  
  def __init__(self, config, global_config, is_extra_msa,a_dim, m_dim, pa_dim):
    super().__init__()
    """Builds EvoformerIteration module.

    Arguments:
      activations: Dictionary containing activations:
        * 'msa': MSA activations, shape [N_seq, N_res, c_m].
        * 'pair': pair activations, shape [N_res, N_res, c_z].
      masks: Dictionary of masks:
        * 'msa': MSA mask, shape [N_seq, N_res].
        * 'pair': pair mask, shape [N_res, N_res].
      is_training: Whether the module is in training mode.
      safe_key: prng.SafeKey encapsulating rng key.

    Returns:
      Outputs, same shape/type as act.
    """
    self.config = config
    self.global_config = global_config
    c = config
    gc = global_config
    self.is_extra_msa = is_extra_msa
    # a_dim = m_dim = 64, pa_dim = 128
    self.msa_row_attention_with_pair_bias = MSARowAttentionWithPairBias(c['msa_row_attention_with_pair_bias'],gc,a_dim, m_dim, pa_dim)
    ### this is real annoucement
    self.msa_column_global_attention = MSAColumnGlobalAttention(c['msa_column_attention'],gc,a_dim,a_dim,a_dim)
    #del self.msa_column_global_attention
    self.msa_transition = Transition(c['msa_transition'], gc,a_dim) # a_dim=64
    # a_dim = 64, pa_dim = 128
    self.outer_product_mean = OuterProductMean(c['outer_product_mean'], gc,a_dim,pa_dim)
    self.triangle_multiplication_outgoing = TriangleMultiplication(c['triangle_multiplication_outgoing'], gc,pa_dim)
    self.triangle_multiplication_incoming = TriangleMultiplication(c['triangle_multiplication_incoming'], gc,pa_dim)
    self.triangle_attention_starting_node = TriangleAttention(c['triangle_attention_starting_node'], gc,pa_dim)
    self.triangle_attention_ending_node = TriangleAttention(c['triangle_attention_ending_node'], gc,pa_dim)
    self.pair_transition = Transition(c['pair_transition'], gc, pa_dim) # pa_dim=128


  def forward(self, msa_act, pair_act, msa_mask, pair_mask):
    msa_act = msa_act + self.msa_row_attention_with_pair_bias(msa_act, msa_mask, pair_act=pair_act) # [TODO] CPU usage is low here
    msa_act = msa_act + self.msa_column_global_attention(msa_act,msa_mask)
    # msa_transition_input.msa_act: torch.Tensor [5120, seq, 64]
    # msa_transition_input.msa_mask: torch.Tensor [5120, seq]
    msa_act = msa_act + self.msa_transition(msa_act,msa_mask)
    # msa_act [5120, seq, 64]
    # msa_mask [5120, seq]
    # pair_act [seq, seq, 128]
    pair_act = pair_act + self.outer_product_mean(msa_act,msa_mask)
    res = {'msa':msa_act}
    pair_act = pair_act + self.triangle_multiplication_outgoing(pair_act,pair_mask)
    pair_act = pair_act + self.triangle_multiplication_incoming(pair_act,pair_mask)
    pair_act = pair_act + self.triangle_attention_starting_node(pair_act,pair_mask)
    pair_act = pair_act + self.triangle_attention_ending_node(pair_act,pair_mask)
    # pair_trainsition_input.pair_act: torch.Tensor [seq, seq, 128]
    # pair_trainsition_input.pair_mask: torch.Tensor [seq, seq]
    pair_act = pair_act + self.pair_transition(pair_act,pair_mask)
    del pair_mask
    res['pair'] = pair_act
    del pair_act
    return res
