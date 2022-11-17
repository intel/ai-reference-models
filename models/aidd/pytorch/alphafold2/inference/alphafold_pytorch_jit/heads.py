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


class MaskedMsaHead(nn.Module):
  """Head to predict MSA at the masked locations.

  The MaskedMsaHead employs a BERT-style objective to reconstruct a masked
  version of the full MSA, based on a linear projection of
  the MSA representation.
  Jumper et al. (2021) Suppl. Sec. 1.9.9 "Masked MSA prediction"
  """

  def __init__(self, config, global_config):
    super().__init__()
    self.config = config
    self.global_config = global_config
    self.logits = nn.Linear(self.config['msa_channel'],self.config['num_output'])

  def __call__(self, representations):
    """Builds MaskedMsaHead module.

    Arguments:
      representations: Dictionary of representations, must contain:
        * 'msa': MSA representation, shape [N_seq, N_res, c_m].
      batch: Batch, unused.
      is_training: Whether the module is in training mode.

    Returns:
      Dictionary containing:
        * 'logits': logits of shape [N_seq, N_res, N_aatype] with
            (unnormalized) log probabilies of predicted aatype at position.
    """
    logits = self.logits(representations['msa'])
    return dict(logits=logits)

  def loss():
    pass
    ### marked! No need!


class DistogramHead(nn.Module):
  """Head to predict a distogram.

  Jumper et al. (2021) Suppl. Sec. 1.9.8 "Distogram prediction"
  """

  def __init__(self, config, global_config, name='distogram_head'):
    super().__init__()
    self.config = config
    self.global_config = global_config
    self.half_logits = nn.Linear(self.config['pair_channel'], self.config['num_bins'])

  def forward(self, representations):
    """Builds DistogramHead module.

    Arguments:
      representations: Dictionary of representations, must contain:
        * 'pair': pair representation, shape [N_res, N_res, c_z].
      batch: Batch, unused.
      is_training: Whether the module is in training mode.

    Returns:
      Dictionary containing:
        * logits: logits for distogram, shape [N_res, N_res, N_bins].
        * bin_breaks: array containing bin breaks, shape [N_bins - 1,].
    """
    half_logits = self.half_logits(representations['pair'])
    logits = half_logits + torch.swapaxes(half_logits, -2, -3)
    breaks = torch.linspace(self.config['first_break'], self.config['last_break'], self.config['num_bins'] - 1)
    return dict(logits=logits, bin_edges=breaks)

  def loss(self, value, batch):
    pass
    # return _distogram_log_loss(value['logits'], value['bin_edges'],
    #                            batch, self.config.num_bins)


class PredictedLDDTHead(nn.Module):
  """Head to predict the per-residue LDDT to be used as a confidence measure.

  Jumper et al. (2021) Suppl. Sec. 1.9.6 "Model confidence prediction (pLDDT)"
  Jumper et al. (2021) Suppl. Alg. 29 "predictPerResidueLDDT_Ca"
  """

  def __init__(self, config, global_config, name='predicted_lddt_head'):
    super().__init__()
    self.config = config
    self.global_config = global_config
    self.input_layer_norm = nn.LayerNorm(normalized_shape=384,elementwise_affine=True)
    # 128 is act.shape[-1]
    self.act_0 = nn.Linear(384,self.config['num_channels'])
    # 128 is act.shape[-1]
    self.act_1 = nn.Linear(self.config['num_channels'], self.config['num_channels'])
    self.logits = nn.Linear(self.config['num_channels'], self.config['num_bins'])

  def forward(self, representations):
    """Builds ExperimentallyResolvedHead module.

    Arguments:
      representations: Dictionary of representations, must contain:
        * 'structure_module': Single representation from the structure module,
             shape [N_res, c_s].
      batch: Batch, unused.
      is_training: Whether the module is in training mode.

    Returns:
      Dictionary containing :
        * 'logits': logits of shape [N_res, N_bins] with
            (unnormalized) log probabilies of binned predicted lDDT.
    """
    act = representations['structure_module']
    act = self.input_layer_norm(act)
    act = F.relu(self.act_0(act))
    act = F.relu(self.act_1(act))
    logits = self.logits(act)
    # Shape (batch_size, num_res, num_bins)
    return dict(logits=logits)

  def loss(self, value, batch):
    ### markder!
    pass

    '''
    
    # Shape (num_res, 37, 3)
    pred_all_atom_pos = value['structure_module']['final_atom_positions']
    # Shape (num_res, 37, 3)
    true_all_atom_pos = batch['all_atom_positions']
    # Shape (num_res, 37)
    all_atom_mask = batch['all_atom_mask']

    # Shape (num_res,)
    lddt_ca = lddt.lddt(
        # Shape (batch_size, num_res, 3)
        predicted_points=pred_all_atom_pos[None, :, 1, :],
        # Shape (batch_size, num_res, 3)
        true_points=true_all_atom_pos[None, :, 1, :],
        # Shape (batch_size, num_res, 1)
        true_points_mask=all_atom_mask[None, :, 1:2].astype(jnp.float32),
        cutoff=15.,
        per_residue=True)[0]
    lddt_ca = jax.lax.stop_gradient(lddt_ca)

    num_bins = self.config.num_bins
    bin_index = jnp.floor(lddt_ca * num_bins).astype(jnp.int32)

    # protect against out of range for lddt_ca == 1
    bin_index = jnp.minimum(bin_index, num_bins - 1)
    lddt_ca_one_hot = jax.nn.one_hot(bin_index, num_classes=num_bins)

    # Shape (num_res, num_channel)
    logits = value['predicted_lddt']['logits']
    errors = softmax_cross_entropy(labels=lddt_ca_one_hot, logits=logits)

    # Shape (num_res,)
    mask_ca = all_atom_mask[:, residue_constants.atom_order['CA']]
    mask_ca = mask_ca.astype(jnp.float32)
    loss = jnp.sum(errors * mask_ca) / (jnp.sum(mask_ca) + 1e-8)

    if self.config.filter_by_resolution:
      # NMR & distillation have resolution = 0
      loss *= ((batch['resolution'] >= self.config.min_resolution)
               & (batch['resolution'] <= self.config.max_resolution)).astype(
                   jnp.float32)

    output = {'loss': loss}
    return output
    '''


class PredictedAlignedErrorHead(nn.Module):
  """Head to predict the distance errors in the backbone alignment frames.

  Can be used to compute predicted TM-Score.
  Jumper et al. (2021) Suppl. Sec. 1.9.7 "TM-score prediction"
  """

  def __init__(self, config, global_config,name='predicted_aligned_error_head'
               ):
    super().__init__() 
    self.config = config
    self.global_config = global_config
    self.logits = nn.Linear(128,self.config['num_bins'])
    # 128 is act.shape[-1]


  def forward(self, representations):
    """Builds PredictedAlignedErrorHead module.

    Arguments:
      representations: Dictionary of representations, must contain:
        * 'pair': pair representation, shape [N_res, N_res, c_z].
      batch: Batch, unused.
      is_training: Whether the module is in training mode.

    Returns:
      Dictionary containing:
        * logits: logits for aligned error, shape [N_res, N_res, N_bins].
        * bin_breaks: array containing bin breaks, shape [N_bins - 1].
    """

    act = representations['pair']

    # Shape (num_res, num_res, num_bins)
    logits = self.logits(act)
    # Shape (num_bins,)
    breaks = torch.linspace(0., self.config['max_error_bin'], self.config['num_bins'] - 1)
    return dict(logits=logits, breaks=breaks)

  def loss(self, value, batch):
    pass
    """
    # Shape (num_res, 7)
    predicted_affine = quat_affine.QuatAffine.from_tensor(
        value['structure_module']['final_affines'])
    # Shape (num_res, 7)
    true_affine = quat_affine.QuatAffine.from_tensor(
        batch['backbone_affine_tensor'])
    # Shape (num_res)
    mask = batch['backbone_affine_mask']
    # Shape (num_res, num_res)
    square_mask = mask[:, None] * mask[None, :]
    num_bins = self.config.num_bins
    # (1, num_bins - 1)
    breaks = value['predicted_aligned_error']['breaks']
    # (1, num_bins)
    logits = value['predicted_aligned_error']['logits']

    # Compute the squared error for each alignment.
    def _local_frame_points(affine):
      points = [jnp.expand_dims(x, axis=-2) for x in affine.translation]
      return affine.invert_point(points, extra_dims=1)
    error_dist2_xyz = [
        jnp.square(a - b)
        for a, b in zip(_local_frame_points(predicted_affine),
                        _local_frame_points(true_affine))]
    error_dist2 = sum(error_dist2_xyz)
    # Shape (num_res, num_res)
    # First num_res are alignment frames, second num_res are the residues.
    error_dist2 = jax.lax.stop_gradient(error_dist2)

    sq_breaks = jnp.square(breaks)
    true_bins = jnp.sum((
        error_dist2[..., None] > sq_breaks).astype(jnp.int32), axis=-1)

    errors = softmax_cross_entropy(
        labels=jax.nn.one_hot(true_bins, num_bins, axis=-1), logits=logits)

    loss = (jnp.sum(errors * square_mask, axis=(-2, -1)) /
            (1e-8 + jnp.sum(square_mask, axis=(-2, -1))))

    if self.config.filter_by_resolution:
      # NMR & distillation have resolution = 0
      loss *= ((batch['resolution'] >= self.config.min_resolution)
               & (batch['resolution'] <= self.config.max_resolution)).astype(
                   jnp.float32)

    output = {'loss': loss}
    return output
    """


class ExperimentallyResolvedHead(nn.Module):
  """Predicts if an atom is experimentally resolved in a high-res structure.

  Only trained on high-resolution X-ray crystals & cryo-EM.
  Jumper et al. (2021) Suppl. Sec. 1.9.10 '"Experimentally resolved" prediction'
  """

  def __init__(self, config, global_config,
               name='experimentally_resolved_head'):
    super().__init__()
    self.config = config
    self.global_config = global_config
    self.logits = nn.Linear(384 ,37)
    # 384 is rep.shape[-1] 37 is atom_exists.shape[-1]
    
  def forward(self, representations):
    """Builds ExperimentallyResolvedHead module.

    Arguments:
      representations: Dictionary of representations, must contain:
        * 'single': Single representation, shape [N_res, c_s].
      batch: Batch, unused.
      is_training: Whether the module is in training mode.

    Returns:
      Dictionary containing:
        * 'logits': logits of shape [N_res, 37],
            log probability that an atom is resolved in atom37 representation,
            can be converted to probability by applying sigmoid.
    """
    logits = self.logits(representations['single'])
    return dict(logits=logits)

  def loss(self, value, batch):
    pass
    '''
    logits = value['logits']
    assert len(logits.shape) == 2

    # Does the atom appear in the amino acid?
    atom_exists = batch['atom37_atom_exists']
    # Is the atom resolved in the experiment? Subset of atom_exists,
    # *except for OXT*
    all_atom_mask = batch['all_atom_mask'].astype(jnp.float32)

    xent = sigmoid_cross_entropy(labels=all_atom_mask, logits=logits)
    loss = jnp.sum(xent * atom_exists) / (1e-8 + jnp.sum(atom_exists))

    if self.config.filter_by_resolution:
      # NMR & distillation examples have resolution = 0.
      loss *= ((batch['resolution'] >= self.config.min_resolution)
               & (batch['resolution'] <= self.config.max_resolution)).astype(
                   jnp.float32)

    output = {'loss': loss}
    return output
    '''
