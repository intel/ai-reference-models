from torch import nn
from torch.nn import functional as F

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

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.poolers import Pooler

from torch.nn import Conv2d


@registry.ROI_KEYPOINT_FEATURE_EXTRACTORS.register("KeypointRCNNFeatureExtractor")
class KeypointRCNNFeatureExtractor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(KeypointRCNNFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        self.pooler = pooler

        input_features = in_channels
        layers = cfg.MODEL.ROI_KEYPOINT_HEAD.CONV_LAYERS
        next_feature = input_features
        self.blocks = []
        for layer_idx, layer_features in enumerate(layers, 1):
            layer_name = "conv_fcn{}".format(layer_idx)
            module = Conv2d(next_feature, layer_features, 3, stride=1, padding=1)
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(module.bias, 0)
            self.add_module(layer_name, module)
            next_feature = layer_features
            self.blocks.append(layer_name)
        self.out_channels = layer_features

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        for layer_name in self.blocks:
            x = F.relu(getattr(self, layer_name)(x))
        return x


def make_roi_keypoint_feature_extractor(cfg, in_channels):
    func = registry.ROI_KEYPOINT_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_KEYPOINT_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg, in_channels)
