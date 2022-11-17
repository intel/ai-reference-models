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
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import pdb
import time


def mask_mean(mask, value, axis=torch.Tensor(), drop_mask_channel=torch.Tensor([False]), eps=torch.Tensor([1e-10])):
  """Masked mean."""
  if drop_mask_channel[0]:
    mask = mask[..., 0]

  mask_shape = mask.shape
  value_shape = value.shape

  assert mask.dim() == value.dim()

  if isinstance(axis, int):
    axis = [axis]
  #elif axis is None:
    #axis = list(range(mask.dim()))
  #assert isinstance(axis, Iterable), ('axis needs to be either an iterable, integer or "None"')

  broadcast_factor = 1.
  for axis_ in axis:
    value_size = value_shape[axis_]
    mask_size = mask_shape[axis_]
    if mask_size == 1: ### [INFO] onnx <torch.Tensor -> bool>: not affect tracing
      broadcast_factor *= value_size
    else:
      if isinstance(mask_size, torch.Tensor) and \
        isinstance(value_size, torch.Tensor):
        assert mask_size.equal(value_size)
      else:
        assert mask_size == value_size
  return (torch.sum(mask * value, dim=int(axis[0])) /
          (torch.sum(mask, dim=int(axis[0])) * broadcast_factor + eps[0]))

def pseudo_beta_fn_with_masks(aatype, all_atom_positions, all_atom_masks):
  """Create pseudo beta features."""
  ### locate glycine (smallest aa)
  is_gly = (aatype == 7)
  ca_idx = 1
  cb_idx = 3
  # is_gly = (aatype == residue_constants.restype_order['G'])
  # ca_idx = residue_constants.atom_order['CA'] # C-alpha
  # cb_idx = residue_constants.atom_order['CB'] # C-beta
  pseudo_beta = torch.where( # [..., None] == hvec2vvec
      torch.tile(is_gly[..., None], [1] * is_gly.dim() + [3]),
      all_atom_positions[..., ca_idx, :],
      all_atom_positions[..., cb_idx, :])


  pseudo_beta_mask = torch.where(
      is_gly, 
      all_atom_masks[..., ca_idx], 
      all_atom_masks[..., cb_idx])
  pseudo_beta_mask = torch.tensor(
    pseudo_beta_mask,
    dtype=torch.float32)
  return pseudo_beta, pseudo_beta_mask
    ### Previous return statement returned a value of type Tuple[Tensor, Tensor] but this return statement returns a value of type Tensor

def pseudo_beta_fn_no_masks(aatype, all_atom_positions):
  """Create pseudo beta features."""
  ### locate glycine (smallest aa)
  is_gly = (aatype == 7)
  ca_idx = 1
  cb_idx = 3
  # is_gly = (aatype == residue_constants.restype_order['G'])
  # ca_idx = residue_constants.atom_order['CA'] # C-alpha
  # cb_idx = residue_constants.atom_order['CB'] # C-beta
  pseudo_beta = torch.where( # [..., None] == hvec2vvec
      torch.tile(is_gly[..., None], [1] * is_gly.dim() + [3]),
      all_atom_positions[..., ca_idx, :],
      all_atom_positions[..., cb_idx, :])
 
  return pseudo_beta  


def dgram_from_positions_pth(positions, num_bins:int, min_bin:float, max_bin:float):
  lower_breaks = torch.linspace(min_bin, max_bin, num_bins, dtype=torch.float32)
  lower_breaks = torch.square(lower_breaks)
  upper_breaks = torch.cat([lower_breaks[1:],torch.tensor([1e8],dtype=torch.float32)], dim=-1)
  dist2 = torch.sum(
    torch.square(
      torch.unsqueeze(positions, dim=-2) - torch.unsqueeze(positions, dim=-3)
    ), dim=-1, keepdim=True
  )
  # Cannot input a tensor of dimension other than 0 as a scalar argument
  disk_lower = dist2 > lower_breaks
  disk_upper = dist2 < upper_breaks
  dgram = disk_lower.float()
  dgram *= disk_upper.float()
  return dgram


def dgram_from_positions(positions, num_bins, min_bin, max_bin):
  """Compute distogram from amino acid positions.

  Arguments:
    positions: [N_res, 3] Position coordinates.
    num_bins: The number of bins in the distogram.
    min_bin: The left edge of the first bin.
    max_bin: The left edge of the final bin. The final bin catches
        everything larger than `max_bin`.

  Returns:
    Distogram with the specified number of bins.
  """

  lower_breaks = np.linspace(min_bin, max_bin, num_bins)
  lower_breaks = np.square(lower_breaks)
  upper_breaks = np.concatenate([lower_breaks[1:],
                                  np.array([1e8], dtype=np.float32)], axis=-1)
  dist2 = np.sum(
    np.square(
      np.expand_dims(positions, axis=-2) - np.expand_dims(positions, axis=-3)
    ),
    axis=-1, keepdims=True)

  dgram = ((dist2 > lower_breaks).astype(np.float32) *
           (dist2 < upper_breaks).astype(np.float32))
  return dgram


def create_extra_msa_feature(extra_msa,
                            extra_has_deletion,
                            extra_deletion_value
                            ):
  """Expand extra_msa into 1hot and concat with other extra msa features.

  We do this as late as possible as the one_hot extra msa can be very large.

  Arguments:
    batch: a dictionary with the following keys:
     * 'extra_msa': [N_extra_seq, N_res] MSA that wasn't selected as a cluster
       centre. Note, that this is not one-hot encoded.
     * 'extra_has_deletion': [N_extra_seq, N_res] Whether there is a deletion to
       the left of each position in the extra MSA.
     * 'extra_deletion_value': [N_extra_seq, N_res] The number of deletions to
       the left of each position in the extra MSA.

  Returns:
    Concatenated tensor of extra MSA features.
  """
  # 23 = 20 amino acids + 'X' for unknown + gap + bert mask
  # [INFO] onnx.Concat node 5, dtypes are consistent now!!!
  msa_1hot = F.one_hot(extra_msa.long(), 23) # torch.int64, [5120, 206, 23]
  msa_feat = [msa_1hot.to(torch.float32),
              torch.unsqueeze(extra_has_deletion, dim=-1), # torch.float32 [5120, 206, 1]
              torch.unsqueeze(extra_deletion_value, dim=-1)] # torch.float32 [5120, 206, 1]
  del msa_1hot
  return torch.cat(msa_feat, dim=-1)


class GlobalAttention(nn.Module):
  """Multihead attention."""

  def __init__(self, config, global_config, a_dim, m_dim, output_dim):
    super().__init__()
    self.config = config
    self.global_config = global_config
    self.output_dim = output_dim
    # k,v dim
    self.key_dim = self.config.get('key_dim', int(a_dim))
    self.value_dim = self.config.get('value_dim', int(m_dim))
    self.num_head = self.config['num_head']
    self.c_gating = self.config['gating']
    assert self.key_dim % self.num_head == 0
    assert self.value_dim % self.num_head == 0
    self.key_dim = self.key_dim // self.num_head
    self.value_dim = self.value_dim // self.num_head
    # q,k,v weights
    self.query_w = nn.Parameter(torch.Tensor(a_dim,self.num_head,self.key_dim))
    self.key_w = nn.Parameter(torch.Tensor(m_dim,self.key_dim))
    self.value_w = nn.Parameter(torch.Tensor(m_dim,self.value_dim))
    self.gating_w = nn.Parameter(torch.Tensor(a_dim,self.num_head,self.value_dim))
    self.gating_b = nn.Parameter(torch.Tensor(self.num_head,self.value_dim))
    self.output_w = nn.Parameter(torch.Tensor(self.num_head,self.value_dim, self.output_dim))
    self.output_b = nn.Parameter(torch.Tensor(self.output_dim))
    # softmax & act fn
    self.softmax = nn.Softmax(dim=-1)
    self.sigmoid = nn.Sigmoid()

  @torch.jit.ignore
  def set_trace(self):
    pdb.set_trace()

  def forward(self, q_data, m_data, q_mask,bias):
    """Builds Attention module.
    Arguments:
      q_data: A tensor of queries, shape [batch_size, N_queries, q_channels].
      m_data: A tensor of memories from which the keys and values are
        projected, shape [batch_size, N_keys, m_channels].
      bias: A bias for the attention, shape [batch_size, N_queries, N_keys].
      nonbatched_bias: Shared bias, shape [N_queries, N_keys].
    Returns:
      A float32 tensor of shape [batch_size, N_queries, output_dim].
    """
    # get query, key, value
    q_avg = mask_mean(q_mask, q_data, axis=torch.tensor([1]))
    q = torch.einsum('ba,ahc->bhc', q_avg, self.query_w) * self.key_dim**(-0.5)
    k = torch.einsum('bka,ac->bkc', m_data, self.key_w)
    v = torch.einsum('bka,ac->bkc', m_data, self.value_w)
    # softmax( query * key ) -> attn matrix
    bias = (1e9 * (q_mask[:, None, :, 0] - 1.))
    logits = torch.einsum('bhc,bkc->bhk', q, k) + bias # [TODO] bias -> ensure it is Tensor
    weights = self.softmax(logits)
    # attn matrix * value -> res
    weighted_avg = torch.einsum('bhk,bkc->bhc', weights, v)  
    # act( linear(q_data) ) * res -> res_gated
    if self.c_gating:
      gate_values = torch.einsum('bqc,chv->bqhv', q_data,self.gating_w) + self.gating_b
      gate_values = self.sigmoid(gate_values)
      weighted_avg = weighted_avg[:, None] * gate_values      # linear(res_gated) -> output
      output = torch.einsum('bqhc,hco->bqo', weighted_avg, self.output_w) + self.output_b
    else:
      output = torch.einsum('bqhc,hco->bqo', weighted_avg, self.output_w) + self.output_b
      output = output[:, None]  
    return output


class MSAColumnGlobalAttention(nn.Module):
  """MSA per-column attention"""

  def __init__(self, config, global_config, a_dim, m_dim, output_dim):
    super().__init__()
    self.config = config
    self.global_config = global_config
    self.query_norm = nn.LayerNorm(normalized_shape = a_dim,
                                  elementwise_affine=True)
    # a_dim=64, m_dim=64, output_dim=64
    self.attention = GlobalAttention(self.config, self.global_config,a_dim, m_dim, output_dim)
    # output_dim = msa_act.shape
    # a_dim = msa_act.shape 
    # m_dim = msa_act.shape
    c = self.config
    assert c['orientation'] == 'per_column'


  def forward(self,
               msa_act, # [206, 5120, 64]
               msa_mask, # pair representation [206, 5120, 1]
               ):
          
    """Builds MSAColumnAttention module.
    Arguments:
      msa_act: [N_seq, N_res, c_m] MSA representation.
      msa_mask: [N_seq, N_res] mask of non-padded regions.
      is_training: Whether the module is in training mode.
    Returns:
      Update to msa_act, shape [N_seq, N_res, c_m]
    """
    assert msa_act.dim() == 3
    assert msa_mask.dim() == 2
    msa_act = torch.swapaxes(msa_act, -2, -3)
    msa_mask = torch.swapaxes(msa_mask, -1, -2)
    bias = (1e9 * (msa_mask - 1.))[:, None, None, :]
    # balance between contribution of msa_act and that of msa_mask
    assert bias.dim() == 4
    msa_act = self.query_norm(msa_act)
    msa_mask = torch.unsqueeze(msa_mask,-1)
    msa_act = self.attention(msa_act, msa_act, msa_mask, bias) # bias [256, 1, 1, 5120]
    msa_act = torch.swapaxes(msa_act, -2, -3)
    return msa_act


class MSAColumnAttention(nn.Module):
  """MSA per-column attention"""

  def __init__(self, config, global_config, a_dim, m_dim, output_dim):
    super().__init__()
    self.config = config
    self.global_config = global_config
    self.query_norm = nn.LayerNorm(normalized_shape = a_dim,
                                  elementwise_affine=True)
    # attention module params: 
    # ['query_w', # [256, 8, 32]
    # 'key_w',    # [256, 8, 32]
    # 'value_w',  
    # 'gating_w', 
    # 'gating_b', 
    # 'output_w', # [8, 32, 256]
    # 'output_b']
    # a_dim=256, m_dim=256, output_dim=256
    # self.config['gating'] is 1, use GatingAttention
    self.attention = GatingAttention(self.config, self.global_config,a_dim, m_dim, output_dim) # 
    # output_dim = msa_act.shape
    # a_dim = msa_act.shape 
    # m_dim = msa_act.shape
    c = self.config
    assert c['orientation'] == 'per_column'
  
  def _slice_attention(self,q_data,m_data,bias,nonbatched_bias=torch.Tensor()):
    """get same result with sliced input."""
    ### avoiding huge memory cost
    ### threhold is ajustable
    threhold = 1000
    unit = 1280 # unit is ajustable
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

  def forward(self,
               msa_act, # torch.Tensor [206, 512, 256]
               msa_mask, # pair representation, torch.Tensor [206, 512]
               ):
          
    """Builds MSAColumnAttention module.
    Arguments:
      msa_act: [N_seq, N_res, c_m] MSA representation.
      msa_mask: [N_seq, N_res] mask of non-padded regions.
      is_training: Whether the module is in training mode.
    Returns:
      Update to msa_act, shape [N_seq, N_res, c_m]
    """
    assert msa_act.dim() == 3
    assert msa_mask.dim() == 2
    msa_act = torch.swapaxes(msa_act, -2, -3)
    msa_mask = torch.swapaxes(msa_mask, -1, -2)
    bias = (1e9 * (msa_mask - 1.))[:, None, None, :] # torch.Tensor [206, 1, 1, 512]
    # balance between contribution of msa_act and that of msa_mask
    assert bias.dim() == 4
    msa_act = self.query_norm(msa_act)
    msa_act = self.attention(msa_act, msa_act, bias)
    msa_act = torch.swapaxes(msa_act, -2, -3)
    return msa_act


class MSARowAttentionWithPairBias(nn.Module):

  def __init__(self,config, global_config, a_dim, m_dim, p_dim):
    super().__init__()
    """
    a_dim = msa_act.shape
    m_dim = msa_mask.shape
    p_dim = pair_act.shape
    """
    self.config = config
    self.global_config = global_config
    self.query_norm = nn.LayerNorm(normalized_shape = a_dim,
                                  elementwise_affine=True)
    self.feat_2d_norm = nn.LayerNorm(normalized_shape = p_dim,
                                  elementwise_affine=True)
    # self.config['gating'] is 1, use GatingAttention
    self.attention = GatingAttention(self.config, self.global_config,a_dim, m_dim, a_dim)
    self.feat_2d_weights = nn.Parameter(torch.Tensor(p_dim,self.config['num_head']))
    assert self.config['orientation'] == 'per_row'


  def _slice_attention(self,q_data,m_data,bias,nonbatched_bias):
    ### avoiding huge memory cost
    ### threhold is ajustable
    threhold = 1000
    unit = 320 # unit is ajustable, 160
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


  def forward(self, 
              msa_act,
              msa_mask,
              pair_act,):
    
    assert msa_act.dim() == 3
    assert msa_mask.dim() == 2
    bias = (1e9 * (msa_mask - 1.))[:, None, None, :]
    msa_act = self.query_norm(msa_act)
    pair_act = self.feat_2d_norm(pair_act)
    nonbatched_bias = torch.einsum('qkc,ch->hqk', pair_act, self.feat_2d_weights)
    msa_act = self._slice_attention(msa_act, msa_act, bias, nonbatched_bias)
    return msa_act

### [done] need rm logic branchs in __init__
# Attention => NoGatingAttention & GatingAttention
# please track config['gating'] to check which module is needed

class NoGatingAttention(nn.Module):
  """Multihead attention w/o Gating"""

  def __init__(self, config, global_config, a_dim, m_dim, output_dim):
    super().__init__()
    self.config = config
    self.global_config = global_config
    self.output_dim = output_dim
    # k,v dim
    self.key_dim = self.config.get('key_dim', int(a_dim))
    self.value_dim = self.config.get('value_dim', int(m_dim))
    self.num_head = self.config['num_head']
    assert self.key_dim % self.num_head == 0
    assert self.value_dim % self.num_head == 0
    self.key_dim = self.key_dim // self.num_head
    self.value_dim = self.value_dim // self.num_head
    # q,k,v weights
    self.query_w = nn.Parameter(torch.Tensor(a_dim,self.num_head,self.key_dim),requires_grad=False)
    self.key_w = nn.Parameter(torch.Tensor(m_dim,self.num_head,self.key_dim),requires_grad=False)
    self.value_w = nn.Parameter(torch.Tensor(m_dim,self.num_head,self.value_dim),requires_grad=False)
    self.output_w = nn.Parameter(torch.Tensor(self.num_head,self.value_dim, self.output_dim),requires_grad=False)
    self.output_b = nn.Parameter(torch.Tensor(self.output_dim),requires_grad=False)
    # softmax & act fn
    self.softmax = nn.Softmax(dim=-1)
    self.sigmoid = nn.Sigmoid()


  @torch.jit.ignore
  def print_op(self, k:str, i:torch.Tensor):
    print('  # [DEBUG] %s =' % k, i.shape, i.stride())

  def forward(self, q_data, m_data, bias, nonbatched_bias=torch.Tensor()):
    """Builds Attention module.
    Arguments:
      q_data: A tensor of queries, shape [batch_size, N_queries, q_channels].
      m_data: A tensor of memories from which the keys and values are
        projected, shape [batch_size, N_keys, m_channels].
      bias: A bias for the attention, shape [batch_size, N_queries, N_keys].
      nonbatched_bias: Shared bias, shape [N_queries, N_keys].
    Returns:
      A float32 tensor of shape [batch_size, N_queries, output_dim].
    """
    # get query, key, value
    q = torch.einsum('bqa,ahc->bqhc', q_data, self.query_w) * self.key_dim**(-0.5)
    k = torch.einsum('bka,ahc->bkhc', m_data, self.key_w)
    v = torch.einsum('bka,ahc->bkhc', m_data, self.value_w)
    # softmax( query * key ) -> attn matrix
    logits = torch.einsum('bqhc,bkhc->bhqk', q, k) + bias
    #if nonbatched_bias is not None:
    if nonbatched_bias.shape[0] > 0:
      logits += torch.unsqueeze(nonbatched_bias, dim=0)
    weights = self.softmax(logits)
    # attn matrix * value -> res
    weighted_avg = torch.einsum('bhqk,bkhc->bqhc', weights, v)
    # linear(res_gated) -> output
    output = torch.einsum('bqhc,hco->bqo', weighted_avg, self.output_w) + self.output_b
    return output


class GatingAttention(nn.Module):
  """Multihead attention w/ Gating"""

  def __init__(self, config, global_config, a_dim, m_dim, output_dim):
    super().__init__()
    self.config = config
    self.global_config = global_config
    self.output_dim = output_dim
    # k,v dim
    self.key_dim = self.config.get('key_dim', int(a_dim))
    self.value_dim = self.config.get('value_dim', int(m_dim))
    self.num_head = self.config['num_head']
    assert self.key_dim % self.num_head == 0
    assert self.value_dim % self.num_head == 0
    self.key_dim = self.key_dim // self.num_head
    self.value_dim = self.value_dim // self.num_head
    # q,k,v weights
    self.query_w = nn.Parameter(torch.Tensor(a_dim,self.num_head,self.key_dim),requires_grad=False)
    self.key_w = nn.Parameter(torch.Tensor(m_dim,self.num_head,self.key_dim),requires_grad=False)
    self.value_w = nn.Parameter(torch.Tensor(m_dim,self.num_head,self.value_dim),requires_grad=False)
    self.gating_w = nn.Parameter(torch.Tensor(a_dim,self.num_head,self.value_dim),requires_grad=False)
    self.gating_b = nn.Parameter(torch.Tensor(self.num_head,self.value_dim),requires_grad=False)
    self.output_w = nn.Parameter(torch.Tensor(self.num_head,self.value_dim, self.output_dim),requires_grad=False)
    self.output_b = nn.Parameter(torch.Tensor(self.output_dim),requires_grad=False)
    # softmax & act fn
    self.softmax = nn.Softmax(dim=-1)
    self.sigmoid = nn.Sigmoid()

  
  @torch.jit.ignore
  def read_time(self) -> float:
    return time.time()


  def forward(self, q_data, m_data, bias, nonbatched_bias=torch.Tensor()):
    """Builds Attention module.
    Arguments:
      q_data: A tensor of queries, shape [batch_size, N_queries, q_channels].
      m_data: A tensor of memories from which the keys and values are
        projected, shape [batch_size, N_keys, m_channels].
      bias: A bias for the attention, shape [batch_size, N_queries, N_keys].
      nonbatched_bias: Shared bias, shape [N_queries, N_keys].
    Returns:
      A float32 tensor of shape [batch_size, N_queries, output_dim].
    """
    # get query, key, value
    q = torch.einsum('bqa,ahc->bqhc', q_data, self.query_w) * self.key_dim**(-0.5)
    k = torch.einsum('bka,ahc->bkhc', m_data, self.key_w)
    v = torch.einsum('bka,ahc->bkhc', m_data, self.value_w)
    #t1 = self.read_time()
    logits = torch.einsum('bqhc,bkhc->bhqk', q, k) + bias
    #t2 = self.read_time()
    #dt = (t2 - t1) *1000
    #print('  #[DEBUG] einsum+add duration =', dt, 'ms')
    #if nonbatched_bias is not None:
    if nonbatched_bias.shape[0] > 0:
      logits += torch.unsqueeze(nonbatched_bias, dim=0)
    weights = self.softmax(logits)
    # attn matrix * value -> res
    weighted_avg = torch.einsum('bhqk,bkhc->bqhc', weights, v)
    # act( linear(q_data) ) * res -> res_gated
    gate_values = torch.einsum('bqc,chv->bqhv', q_data,self.gating_w) + self.gating_b
    gate_values = self.sigmoid(gate_values)
    weighted_avg *= gate_values
    # linear(res_gated) -> output
    output = torch.einsum('bqhc,hco->bqo', weighted_avg, self.output_w) + self.output_b
    return output

