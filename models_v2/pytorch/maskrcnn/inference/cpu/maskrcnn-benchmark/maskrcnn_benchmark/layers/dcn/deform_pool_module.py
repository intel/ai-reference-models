from torch import nn

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
from .deform_pool_func import deform_roi_pooling


class DeformRoIPooling(nn.Module):

    def __init__(self,
                 spatial_scale,
                 out_size,
                 out_channels,
                 no_trans,
                 group_size=1,
                 part_size=None,
                 sample_per_part=4,
                 trans_std=.0):
        super(DeformRoIPooling, self).__init__()
        self.spatial_scale = spatial_scale
        self.out_size = out_size
        self.out_channels = out_channels
        self.no_trans = no_trans
        self.group_size = group_size
        self.part_size = out_size if part_size is None else part_size
        self.sample_per_part = sample_per_part
        self.trans_std = trans_std

    def forward(self, data, rois, offset):
        if self.no_trans:
            offset = data.new_empty(0)
        return deform_roi_pooling(
            data, rois, offset, self.spatial_scale, self.out_size,
            self.out_channels, self.no_trans, self.group_size, self.part_size,
            self.sample_per_part, self.trans_std)


class DeformRoIPoolingPack(DeformRoIPooling):

    def __init__(self,
                 spatial_scale,
                 out_size,
                 out_channels,
                 no_trans,
                 group_size=1,
                 part_size=None,
                 sample_per_part=4,
                 trans_std=.0,
                 deform_fc_channels=1024):
        super(DeformRoIPoolingPack,
              self).__init__(spatial_scale, out_size, out_channels, no_trans,
                             group_size, part_size, sample_per_part, trans_std)

        self.deform_fc_channels = deform_fc_channels

        if not no_trans:
            self.offset_fc = nn.Sequential(
                nn.Linear(self.out_size * self.out_size * self.out_channels,
                          self.deform_fc_channels),
                nn.ReLU(inplace=True),
                nn.Linear(self.deform_fc_channels, self.deform_fc_channels),
                nn.ReLU(inplace=True),
                nn.Linear(self.deform_fc_channels,
                          self.out_size * self.out_size * 2))
            self.offset_fc[-1].weight.data.zero_()
            self.offset_fc[-1].bias.data.zero_()

    def forward(self, data, rois):
        assert data.size(1) == self.out_channels
        if self.no_trans:
            offset = data.new_empty(0)
            return deform_roi_pooling(
                data, rois, offset, self.spatial_scale, self.out_size,
                self.out_channels, self.no_trans, self.group_size,
                self.part_size, self.sample_per_part, self.trans_std)
        else:
            n = rois.shape[0]
            offset = data.new_empty(0)
            x = deform_roi_pooling(data, rois, offset, self.spatial_scale,
                                   self.out_size, self.out_channels, True,
                                   self.group_size, self.part_size,
                                   self.sample_per_part, self.trans_std)
            offset = self.offset_fc(x.view(n, -1))
            offset = offset.view(n, 2, self.out_size, self.out_size)
            return deform_roi_pooling(
                data, rois, offset, self.spatial_scale, self.out_size,
                self.out_channels, self.no_trans, self.group_size,
                self.part_size, self.sample_per_part, self.trans_std)


class ModulatedDeformRoIPoolingPack(DeformRoIPooling):

    def __init__(self,
                 spatial_scale,
                 out_size,
                 out_channels,
                 no_trans,
                 group_size=1,
                 part_size=None,
                 sample_per_part=4,
                 trans_std=.0,
                 deform_fc_channels=1024):
        super(ModulatedDeformRoIPoolingPack, self).__init__(
            spatial_scale, out_size, out_channels, no_trans, group_size,
            part_size, sample_per_part, trans_std)

        self.deform_fc_channels = deform_fc_channels

        if not no_trans:
            self.offset_fc = nn.Sequential(
                nn.Linear(self.out_size * self.out_size * self.out_channels,
                          self.deform_fc_channels),
                nn.ReLU(inplace=True),
                nn.Linear(self.deform_fc_channels, self.deform_fc_channels),
                nn.ReLU(inplace=True),
                nn.Linear(self.deform_fc_channels,
                          self.out_size * self.out_size * 2))
            self.offset_fc[-1].weight.data.zero_()
            self.offset_fc[-1].bias.data.zero_()
            self.mask_fc = nn.Sequential(
                nn.Linear(self.out_size * self.out_size * self.out_channels,
                          self.deform_fc_channels),
                nn.ReLU(inplace=True),
                nn.Linear(self.deform_fc_channels,
                          self.out_size * self.out_size * 1),
                nn.Sigmoid())
            self.mask_fc[2].weight.data.zero_()
            self.mask_fc[2].bias.data.zero_()

    def forward(self, data, rois):
        assert data.size(1) == self.out_channels
        if self.no_trans:
            offset = data.new_empty(0)
            return deform_roi_pooling(
                data, rois, offset, self.spatial_scale, self.out_size,
                self.out_channels, self.no_trans, self.group_size,
                self.part_size, self.sample_per_part, self.trans_std)
        else:
            n = rois.shape[0]
            offset = data.new_empty(0)
            x = deform_roi_pooling(data, rois, offset, self.spatial_scale,
                                   self.out_size, self.out_channels, True,
                                   self.group_size, self.part_size,
                                   self.sample_per_part, self.trans_std)
            offset = self.offset_fc(x.view(n, -1))
            offset = offset.view(n, 2, self.out_size, self.out_size)
            mask = self.mask_fc(x.view(n, -1))
            mask = mask.view(n, 1, self.out_size, self.out_size)
            return deform_roi_pooling(
                data, rois, offset, self.spatial_scale, self.out_size,
                self.out_channels, self.no_trans, self.group_size,
                self.part_size, self.sample_per_part, self.trans_std) * mask
