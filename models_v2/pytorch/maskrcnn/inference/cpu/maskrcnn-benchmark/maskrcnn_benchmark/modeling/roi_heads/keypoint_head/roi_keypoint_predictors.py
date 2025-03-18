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
from maskrcnn_benchmark import layers
from maskrcnn_benchmark.modeling import registry


@registry.ROI_KEYPOINT_PREDICTOR.register("KeypointRCNNPredictor")
class KeypointRCNNPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(KeypointRCNNPredictor, self).__init__()
        input_features = in_channels
        num_keypoints = cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_CLASSES
        deconv_kernel = 4
        self.kps_score_lowres = layers.ConvTranspose2d(
            input_features,
            num_keypoints,
            deconv_kernel,
            stride=2,
            padding=deconv_kernel // 2 - 1,
        )
        nn.init.kaiming_normal_(
            self.kps_score_lowres.weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.constant_(self.kps_score_lowres.bias, 0)
        self.up_scale = 2
        self.out_channels = num_keypoints

    def forward(self, x):
        x = self.kps_score_lowres(x)
        x = layers.interpolate(
            x, scale_factor=self.up_scale, mode="bilinear", align_corners=False
        )
        return x


def make_roi_keypoint_predictor(cfg, in_channels):
    func = registry.ROI_KEYPOINT_PREDICTOR[cfg.MODEL.ROI_KEYPOINT_HEAD.PREDICTOR]
    return func(cfg, in_channels)
