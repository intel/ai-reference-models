# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np
from typing import Tuple
import functools
import jax

# pylint: disable=bad-whitespace
QUAT_TO_ROT = np.zeros((4, 4, 3, 3), dtype=np.float32)

QUAT_TO_ROT[0, 0] = [[ 1, 0, 0], [ 0, 1, 0], [ 0, 0, 1]]  # rr
QUAT_TO_ROT[1, 1] = [[ 1, 0, 0], [ 0,-1, 0], [ 0, 0,-1]]  # ii
QUAT_TO_ROT[2, 2] = [[-1, 0, 0], [ 0, 1, 0], [ 0, 0,-1]]  # jj
QUAT_TO_ROT[3, 3] = [[-1, 0, 0], [ 0,-1, 0], [ 0, 0, 1]]  # kk

QUAT_TO_ROT[1, 2] = [[ 0, 2, 0], [ 2, 0, 0], [ 0, 0, 0]]  # ij
QUAT_TO_ROT[1, 3] = [[ 0, 0, 2], [ 0, 0, 0], [ 2, 0, 0]]  # ik
QUAT_TO_ROT[2, 3] = [[ 0, 0, 0], [ 0, 0, 2], [ 0, 2, 0]]  # jk

QUAT_TO_ROT[0, 1] = [[ 0, 0, 0], [ 0, 0,-2], [ 0, 2, 0]]  # ir
QUAT_TO_ROT[0, 2] = [[ 0, 0, 2], [ 0, 0, 0], [-2, 0, 0]]  # jr
QUAT_TO_ROT[0, 3] = [[ 0,-2, 0], [ 2, 0, 0], [ 0, 0, 0]]  # kr

QUAT_MULTIPLY = np.zeros((4, 4, 4), dtype=np.float32)
QUAT_MULTIPLY[:, :, 0] = [[ 1, 0, 0, 0],
                          [ 0,-1, 0, 0],
                          [ 0, 0,-1, 0],
                          [ 0, 0, 0,-1]]

QUAT_MULTIPLY[:, :, 1] = [[ 0, 1, 0, 0],
                          [ 1, 0, 0, 0],
                          [ 0, 0, 0, 1],
                          [ 0, 0,-1, 0]]

QUAT_MULTIPLY[:, :, 2] = [[ 0, 0, 1, 0],
                          [ 0, 0, 0,-1],
                          [ 1, 0, 0, 0],
                          [ 0, 1, 0, 0]]

QUAT_MULTIPLY[:, :, 3] = [[ 0, 0, 0, 1],
                          [ 0, 0, 1, 0],
                          [ 0,-1, 0, 0],
                          [ 1, 0, 0, 0]]

QUAT_MULTIPLY_BY_VEC = QUAT_MULTIPLY[:, 1:, :]

def quat_to_rot_pth(normalized_quat):
  """Convert a normalized quaternion to a rotation matrix."""
  rot_tensor = torch.sum(
      torch.reshape((QUAT_TO_ROT), (4, 4, 9)) *
      normalized_quat[..., :, None, None] *
      normalized_quat[..., None, :, None],
      axis=(-3, -2))
  rot = torch.moveaxis(rot_tensor, -1, 0)  # Unstack.
  return torch.stack(
          torch.stack(rot[0], rot[1], rot[2]),
          torch.stack(rot[3], rot[4], rot[5]),
          torch.stack(rot[6], rot[7], rot[8]))

def quat_to_rot(normalized_quat):
  """Convert a normalized quaternion to a rotation matrix."""
  rot_tensor = np.sum(
      np.reshape((QUAT_TO_ROT), (4, 4, 9)) *
      normalized_quat[..., :, None, None] *
      normalized_quat[..., None, :, None],
      axis=(-3, -2))
  rot = np.moveaxis(rot_tensor, -1, 0)  # Unstack.
  return [[rot[0], rot[1], rot[2]],
          [rot[3], rot[4], rot[5]],
          [rot[6], rot[7], rot[8]]]


def apply_inverse_rot_to_vec(rot, vec):
  """Multiply the inverse of a rotation matrix by a vector."""
  # Inverse rotation is just transpose
  return [rot[0][0] * vec[0] + rot[1][0] * vec[1] + rot[2][0] * vec[2],
          rot[0][1] * vec[0] + rot[1][1] * vec[1] + rot[2][1] * vec[2],
          rot[0][2] * vec[0] + rot[1][2] * vec[1] + rot[2][2] * vec[2]]


def _multiply(a, b):
  return np.stack([
      np.array([a[0][0]*b[0][0] + a[0][1]*b[1][0] + a[0][2]*b[2][0],
                 a[0][0]*b[0][1] + a[0][1]*b[1][1] + a[0][2]*b[2][1],
                 a[0][0]*b[0][2] + a[0][1]*b[1][2] + a[0][2]*b[2][2]]),

      np.array([a[1][0]*b[0][0] + a[1][1]*b[1][0] + a[1][2]*b[2][0],
                 a[1][0]*b[0][1] + a[1][1]*b[1][1] + a[1][2]*b[2][1],
                 a[1][0]*b[0][2] + a[1][1]*b[1][2] + a[1][2]*b[2][2]]),

      np.array([a[2][0]*b[0][0] + a[2][1]*b[1][0] + a[2][2]*b[2][0],
                 a[2][0]*b[0][1] + a[2][1]*b[1][1] + a[2][2]*b[2][1],
                 a[2][0]*b[0][2] + a[2][1]*b[1][2] + a[2][2]*b[2][2]])])

def _multiply_pth(a, b):
  return torch.stack([
    torch.stack([a[0][0]*b[0][0] + a[0][1]*b[1][0] + a[0][2]*b[2][0],
                  a[0][0]*b[0][1] + a[0][1]*b[1][1] + a[0][2]*b[2][1],
                  a[0][0]*b[0][2] + a[0][1]*b[1][2] + a[0][2]*b[2][2]]),
    torch.stack([a[1][0]*b[0][0] + a[1][1]*b[1][0] + a[1][2]*b[2][0],
                  a[1][0]*b[0][1] + a[1][1]*b[1][1] + a[1][2]*b[2][1],
                  a[1][0]*b[0][2] + a[1][1]*b[1][2] + a[1][2]*b[2][2]]),
    torch.stack([a[2][0]*b[0][0] + a[2][1]*b[1][0] + a[2][2]*b[2][0],
                  a[2][0]*b[0][1] + a[2][1]*b[1][1] + a[2][2]*b[2][1],
                  a[2][0]*b[0][2] + a[2][1]*b[1][2] + a[2][2]*b[2][2]])
  ])

def apply_rot_to_vec(rot, vec, unstack=False):
  """Multiply rotation matrix by a vector."""
  if unstack:
    x, y, z = [vec[:, i] for i in range(3)]
  else:
    x, y, z = vec
  return [rot[0][0] * x + rot[0][1] * y + rot[0][2] * z,
          rot[1][0] * x + rot[1][1] * y + rot[1][2] * z,
          rot[2][0] * x + rot[2][1] * y + rot[2][2] * z]

def apply_rot_to_vec_pth(rot, vec, unstack:bool=False):
  """Multiply rotation matrix by a vector."""
  if unstack:
    x, y, z = [vec[:, i] for i in range(3)]
  else:
    x, y, z = vec[0], vec[1], vec[2]
  return [rot[0][0] * x + rot[0][1] * y + rot[0][2] * z,
          rot[1][0] * x + rot[1][1] * y + rot[1][2] * z,
          rot[2][0] * x + rot[2][1] * y + rot[2][2] * z]

def make_canonical_transform_pth(n_xyz, ca_xyz, c_xyz):
  assert len(n_xyz.shape) == 2, n_xyz.shape
  assert n_xyz.shape[-1] == 3, n_xyz.shape
  assert n_xyz.shape == ca_xyz.shape == c_xyz.shape, (
      n_xyz.shape, ca_xyz.shape, c_xyz.shape)

  # Place CA at the origin.
  translation = -ca_xyz
  n_xyz = n_xyz + translation
  c_xyz = c_xyz + translation

  # Place C on the x-axis.
  c_x, c_y, c_z = [ c_xyz[:,i] for i in range(3) ] 
  # Rotate by angle c1 in the x-y plane (around the z-axis).
  sin_c1 = -c_y / torch.sqrt(1e-20 + c_x**2 + c_y**2)
  cos_c1 = c_x / torch.sqrt(1e-20 + c_x**2 + c_y**2)
  zeros = torch.zeros_like(sin_c1)
  ones = torch.ones_like(sin_c1)
  # pylint: disable=bad-whitespace
  c1_rot_matrix = torch.stack([torch.stack([cos_c1, -sin_c1, zeros]),
                               torch.stack([sin_c1,  cos_c1, zeros]),
                               torch.stack([zeros,    zeros,  ones])])

  # Rotate by angle c2 in the x-z plane (around the y-axis).
  sin_c2 = c_z / torch.sqrt(1e-20 + c_x**2 + c_y**2 + c_z**2)
  cos_c2 = torch.sqrt(c_x**2 + c_y**2) / torch.sqrt(
      1e-20 + c_x**2 + c_y**2 + c_z**2)
  c2_rot_matrix = torch.stack([torch.stack([cos_c2,  zeros, sin_c2]),
                               torch.stack([zeros,    ones,  zeros]),
                               torch.stack([-sin_c2, zeros, cos_c2])])

  c_rot_matrix = _multiply_pth(c2_rot_matrix, c1_rot_matrix)
  n_xyz = torch.stack(apply_rot_to_vec_pth(c_rot_matrix, n_xyz, unstack=True)).T

  # Place N in the x-y plane.
  _, n_y, n_z = [n_xyz[:, i] for i in range(3)]
  # Rotate by angle alpha in the y-z plane (around the x-axis).
  sin_n = -n_z / torch.sqrt(1e-20 + n_y**2 + n_z**2)
  cos_n =  n_y / torch.sqrt(1e-20 + n_y**2 + n_z**2)
  n_rot_matrix = torch.stack([torch.stack([ones,  zeros,  zeros]),
                              torch.stack([zeros, cos_n, -sin_n]),
                              torch.stack([zeros, sin_n,  cos_n])])
  # pylint: enable=bad-whitespace

  return (translation,
          torch.permute(_multiply_pth(n_rot_matrix, c_rot_matrix), [2, 0, 1]))

def make_canonical_transform(
    n_xyz: np.ndarray,
    ca_xyz: np.ndarray,
    c_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Returns translation and rotation matrices to canonicalize residue atoms.

  Note that this method does not take care of symmetries. If you provide the
  atom positions in the non-standard way, the N atom will end up not at
  [-0.527250, 1.359329, 0.0] but instead at [-0.527250, -1.359329, 0.0]. You
  need to take care of such cases in your code.

  Args:
    n_xyz: An array of shape [batch, 3] of nitrogen xyz coordinates.
    ca_xyz: An array of shape [batch, 3] of carbon alpha xyz coordinates.
    c_xyz: An array of shape [batch, 3] of carbon xyz coordinates.

  Returns:
    A tuple (translation, rotation) where:
      translation is an array of shape [batch, 3] defining the translation.
      rotation is an array of shape [batch, 3, 3] defining the rotation.
    After applying the translation and rotation to all atoms in a residue:
      * All atoms will be shifted so that CA is at the origin,
      * All atoms will be rotated so that C is at the x-axis,
      * All atoms will be shifted so that N is in the xy plane.
  """
  assert len(n_xyz.shape) == 2, n_xyz.shape
  assert n_xyz.shape[-1] == 3, n_xyz.shape
  assert n_xyz.shape == ca_xyz.shape == c_xyz.shape, (
      n_xyz.shape, ca_xyz.shape, c_xyz.shape)

  # Place CA at the origin.
  translation = -ca_xyz
  n_xyz = n_xyz + translation
  c_xyz = c_xyz + translation

  # Place C on the x-axis.
  c_x, c_y, c_z = [ c_xyz[:,i] for i in range(3) ] 
  # Rotate by angle c1 in the x-y plane (around the z-axis).
  sin_c1 = -c_y / np.sqrt(1e-20 + c_x**2 + c_y**2)
  cos_c1 = c_x / np.sqrt(1e-20 + c_x**2 + c_y**2)
  zeros = np.zeros_like(sin_c1)
  ones = np.ones_like(sin_c1)
  # pylint: disable=bad-whitespace
  c1_rot_matrix = np.stack([np.array([cos_c1, -sin_c1, zeros]),
                             np.array([sin_c1,  cos_c1, zeros]),
                             np.array([zeros,    zeros,  ones])])

  # Rotate by angle c2 in the x-z plane (around the y-axis).
  sin_c2 = c_z / np.sqrt(1e-20 + c_x**2 + c_y**2 + c_z**2)
  cos_c2 = np.sqrt(c_x**2 + c_y**2) / np.sqrt(
      1e-20 + c_x**2 + c_y**2 + c_z**2)
  c2_rot_matrix = np.stack([np.array([cos_c2,  zeros, sin_c2]),
                             np.array([zeros,    ones,  zeros]),
                             np.array([-sin_c2, zeros, cos_c2])])

  c_rot_matrix = _multiply(c2_rot_matrix, c1_rot_matrix)
  n_xyz = np.stack(apply_rot_to_vec(c_rot_matrix, n_xyz, unstack=True)).T

  # Place N in the x-y plane.
  _, n_y, n_z = [n_xyz[:, i] for i in range(3)]
  # Rotate by angle alpha in the y-z plane (around the x-axis).
  sin_n = -n_z / np.sqrt(1e-20 + n_y**2 + n_z**2)
  cos_n = n_y / np.sqrt(1e-20 + n_y**2 + n_z**2)
  n_rot_matrix = np.stack([np.array([ones,  zeros,  zeros]),
                            np.array([zeros, cos_n, -sin_n]),
                            np.array([zeros, sin_n,  cos_n])])
  # pylint: enable=bad-whitespace

  return (translation,
          np.transpose(_multiply(n_rot_matrix, c_rot_matrix), [2, 0, 1]))

# @in_pth2np_out_np2pth
def make_transform_from_reference(
    n_xyz: torch.Tensor,
    ca_xyz: torch.Tensor,
    c_xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
  """Returns rotation and translation matrices to convert from reference.

  Note that this method does not take care of symmetries. If you provide the
  atom positions in the non-standard way, the N atom will end up not at
  [-0.527250, 1.359329, 0.0] but instead at [-0.527250, -1.359329, 0.0]. You
  need to take care of such cases in your code.

  Args:
    n_xyz: An array of shape [batch, 3] of nitrogen xyz coordinates.
    ca_xyz: An array of shape [batch, 3] of carbon alpha xyz coordinates.
    c_xyz: An array of shape [batch, 3] of carbon xyz coordinates.

  Returns:
    A tuple (rotation, translation) where:
      rotation is an array of shape [batch, 3, 3] defining the rotation.
      translation is an array of shape [batch, 3] defining the translation.
    After applying the translation and rotation to the reference backbone,
    the coordinates will approximately equal to the input coordinates.

    The order of translation and rotation differs from make_canonical_transform
    because the rotation from this function should be applied before the
    translation, unlike make_canonical_transform.
  """
  translation, rotation = make_canonical_transform(n_xyz, ca_xyz, c_xyz)
  return np.transpose(rotation, [0, 2, 1]), -translation

def make_transform_from_reference_pth(n_xyz,ca_xyz,c_xyz):
  translation, rotation = make_canonical_transform_pth(n_xyz, ca_xyz, c_xyz)
  return torch.permute(rotation, [0, 2, 1]), -translation

def quat_multiply_by_vec(quat, vec):
  """Multiply a quaternion by a pure-vector quaternion."""
  return np.sum(
      QUAT_MULTIPLY_BY_VEC *
      quat[..., :, None, None] *
      vec[..., None, :, None],
      axis=(-3, -2))

def quat_multiply_by_vec_pth(quat, vec):
  return torch.sum(
    QUAT_MULTIPLY_BY_VEC *
    quat[..., :, None, None] *
    vec[..., None, :, None],
    axis=(-3, -2))

def rot_to_quat(rot, unstack_inputs=False):
  """Convert rotation matrix to quaternion.

  Note that this function calls self_adjoint_eig which is extremely expensive on
  the GPU. If at all possible, this function should run on the CPU.

  Args:
     rot: rotation matrix (see below for format).
     unstack_inputs:  If true, rotation matrix should be shape (..., 3, 3)
       otherwise the rotation matrix should be a list of lists of tensors.

  Returns:
    Quaternion as (..., 4) tensor.
  """
  if unstack_inputs:
    rot = [np.moveaxis(x, -1, 0) for x in np.moveaxis(rot, -2, 0)]

  [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = rot

  # pylint: disable=bad-whitespace
  k = [[ xx + yy + zz,      zy - yz,      xz - zx,      yx - xy,],
       [      zy - yz, xx - yy - zz,      xy + yx,      xz + zx,],
       [      xz - zx,      xy + yx, yy - xx - zz,      yz + zy,],
       [      yx - xy,      xz + zx,      yz + zy, zz - xx - yy,]]
  # pylint: enable=bad-whitespace

  k = (1./3.) * np.stack([np.stack(x, axis=-1) for x in k],
                          axis=-2)

  # Get eigenvalues in non-decreasing order and associated.
  _, qs = np.linalg.eigh(k)
  return qs[..., -1]

def rot_to_quat_pth(rot, unstack_inputs=False):
  if unstack_inputs:
    rot = [torch.moveaxis(x, -1, 0) for x in torch.moveaxis(rot, -2, 0)]
  [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = rot
  # pylint: disable=bad-whitespace
  k = [[ xx + yy + zz,      zy - yz,      xz - zx,      yx - xy,],
       [      zy - yz, xx - yy - zz,      xy + yx,      xz + zx,],
       [      xz - zx,      xy + yx, yy - xx - zz,      yz + zy,],
       [      yx - xy,      xz + zx,      yz + zy, zz - xx - yy,]]
  # pylint: enable=bad-whitespace
  k = (1./3.) * torch.stack([torch.stack(x, axis=-1) for x in k], axis=-2)
  # Get eigenvalues in non-decreasing order and associated.
  _, qs = torch.linalg.eigh(k)
  return qs[..., -1]


class QuatAffine(object):
  """Affine transformation represented by quaternion and vector."""
  def __init__(self, quaternion, translation, rotation=None, normalize=True,
               unstack_inputs=False):
    """Initialize from quaternion and translation.

    Args:
      quaternion: Rotation represented by a quaternion, to be applied
        before translation.  Must be a unit quaternion unless normalize==True.
      translation: Translation represented as a vector.
      rotation: Same rotation as the quaternion, represented as a (..., 3, 3)
        tensor.  If None, rotation will be calculated from the quaternion.
      normalize: If True, l2 normalize the quaternion on input.
      unstack_inputs: If True, translation is a vector with last component 3
    """

    if quaternion is not None:
      assert quaternion.shape[-1] == 4

    if unstack_inputs:
      if rotation is not None:
        rotation = [np.moveaxis(x, -1, 0)   # Unstack.
                    for x in np.moveaxis(rotation, -2, 0)]  # Unstack.
      translation = np.moveaxis(translation, -1, 0)  # Unstack.

    if normalize and quaternion is not None:
      quaternion = quaternion / np.linalg.norm(quaternion, axis=-1,
                                                keepdims=True)

    if rotation is None:
      rotation = quat_to_rot(quaternion)

    self.quaternion = quaternion
    self.rotation = [list(row) for row in rotation]
    self.translation = list(translation)

    assert all(len(row) == 3 for row in self.rotation)
    assert len(self.translation) == 3

  def to_tensor(self):
    return np.cat(
        [self.quaternion] +
        [np.unsqueeze(x, axis=-1) for x in self.translation],
        axis=-1)

  def apply_tensor_fn(self, tensor_fn):
    """Return a new QuatAffine with tensor_fn applied (e.g. stop_gradient)."""
    return QuatAffine(
        tensor_fn(self.quaternion),
        [tensor_fn(x) for x in self.translation],
        rotation=[[tensor_fn(x) for x in row] for row in self.rotation],
        normalize=False)

  def apply_rotation_tensor_fn(self, tensor_fn):
    """Return a new QuatAffine with tensor_fn applied to the rotation part."""
    return QuatAffine(
        tensor_fn(self.quaternion),
        [x for x in self.translation],
        rotation=[[tensor_fn(x) for x in row] for row in self.rotation],
        normalize=False)

  def apply_rotation_stop_gradient(self):
    return QuatAffine(
        jax.lax.stop_gradient(self.quaternion),
        [x for x in self.translation],
        rotation=[[jax.lax.stop_gradient(x) for x in row] for row in self.rotation],
        normalize=False)

  def scale_translation(self, position_scale):
    """Return a new quat affine with a different scale for translation."""

    return QuatAffine(
        self.quaternion,
        [x * position_scale for x in self.translation],
        rotation=[[x for x in row] for row in self.rotation],
        normalize=False)

  @classmethod
  def from_tensor(cls, tensor, normalize=False):
    quaternion, tx, ty, tz = np.split(tensor, [4, 5, 6], axis=-1)
    return cls(quaternion,
               [tx[..., 0], ty[..., 0], tz[..., 0]],
               normalize=normalize)

  def pre_compose(self, update):
    """Return a new QuatAffine which applies the transformation update first.

    Args:
      update: Length-6 vector. 3-vector of x, y, and z such that the quaternion
        update is (1, x, y, z) and zero for the 3-vector is the identity
        quaternion. 3-vector for translation concatenated.

    Returns:
      New QuatAffine object.
    """
    vector_quaternion_update, x, y, z = np.split(update, [3, 4, 5], axis=-1)
    trans_update = [np.squeeze(x, axis=-1),
                    np.squeeze(y, axis=-1),
                    np.squeeze(z, axis=-1)]

    new_quaternion = (self.quaternion +
                      quat_multiply_by_vec(self.quaternion,
                                           vector_quaternion_update))

    trans_update = apply_rot_to_vec_pth(self.rotation, trans_update)
    new_translation = [
        self.translation[0] + trans_update[0],
        self.translation[1] + trans_update[1],
        self.translation[2] + trans_update[2]]

    return QuatAffine(new_quaternion, new_translation)

  def apply_to_point(self, point, extra_dims=0):
    """Apply affine to a point.

    Args:
      point: List of 3 tensors to apply affine.
      extra_dims:  Number of dimensions at the end of the transformed_point
        shape that are not present in the rotation and translation.  The most
        common use is rotation N points at once with extra_dims=1 for use in a
        network.

    Returns:
      Transformed point after applying affine.
    """
    rotation = self.rotation
    translation = self.translation
    for _ in range(extra_dims):
      expand_fn = functools.partial(np.squeeze, axis=-1)
      rotation = map(expand_fn, rotation)
      translation = map(expand_fn, translation)

    rot_point = apply_rot_to_vec(rotation, point)
    return [
        rot_point[0] + translation[0],
        rot_point[1] + translation[1],
        rot_point[2] + translation[2]]

  def invert_point(self, transformed_point, extra_dims=0):
    """Apply inverse of transformation to a point.

    Args:
      transformed_point: List of 3 tensors to apply affine
      extra_dims:  Number of dimensions at the end of the transformed_point
        shape that are not present in the rotation and translation.  The most
        common use is rotation N points at once with extra_dims=1 for use in a
        network.

    Returns:
      Transformed point after applying affine.
    """
    rotation = self.rotation
    translation = self.translation
    for _ in range(extra_dims):
      expand_fn = functools.partial(np.expand_dims, axis=-1)
      rotation = expand_fn(rotation)
      translation = expand_fn(translation)

    rot_point = [
        transformed_point[0] - translation[0],
        transformed_point[1] - translation[1],
        transformed_point[2] - translation[2]]

    return apply_inverse_rot_to_vec(rotation, rot_point)

  def __repr__(self):
    return 'QuatAffine(%r, %r)' % (self.quaternion, self.translation)

