# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

# MIT License
#
# Copyright (c) 2018 Facebook
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from .batch_norm import FrozenBatchNorm2d
from .misc import Conv2d
from .misc import DFConv2d
from .misc import ConvTranspose2d
from .misc import BatchNorm2d
from .misc import interpolate
from .nms import nms
from .roi_align import ROIAlign
from .roi_align import roi_align
from .roi_pool import ROIPool
from .roi_pool import roi_pool
from .smooth_l1_loss import smooth_l1_loss
from .sigmoid_focal_loss import SigmoidFocalLoss
from .dcn.deform_conv_func import deform_conv, modulated_deform_conv
from .dcn.deform_conv_module import (
    DeformConv,
    ModulatedDeformConv,
    ModulatedDeformConvPack,
)
from .dcn.deform_pool_func import deform_roi_pooling
from .dcn.deform_pool_module import (
    DeformRoIPooling,
    DeformRoIPoolingPack,
    ModulatedDeformRoIPoolingPack,
)


__all__ = [
    "nms",
    "roi_align",
    "ROIAlign",
    "roi_pool",
    "ROIPool",
    "smooth_l1_loss",
    "Conv2d",
    "DFConv2d",
    "ConvTranspose2d",
    "interpolate",
    "BatchNorm2d",
    "FrozenBatchNorm2d",
    "SigmoidFocalLoss",
    "deform_conv",
    "modulated_deform_conv",
    "DeformConv",
    "ModulatedDeformConv",
    "ModulatedDeformConvPack",
    "deform_roi_pooling",
    "DeformRoIPooling",
    "DeformRoIPoolingPack",
    "ModulatedDeformRoIPoolingPack",
]
