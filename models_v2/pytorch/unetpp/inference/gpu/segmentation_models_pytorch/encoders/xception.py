#
#The MIT License
#
#Copyright (c) 2019, Pavel Iakubovskii
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in
#all copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#THE SOFTWARE.
#

import re
import torch.nn as nn

from pretrainedmodels.models.xception import pretrained_settings
from pretrainedmodels.models.xception import Xception

from ._base import EncoderMixin


class XceptionEncoder(Xception, EncoderMixin):
    def __init__(self, out_channels, *args, depth=5, **kwargs):
        super().__init__(*args, **kwargs)

        self._out_channels = out_channels
        self._depth = depth
        self._in_channels = 3

        # modify padding to maintain output shape
        self.conv1.padding = (1, 1)
        self.conv2.padding = (1, 1)

        del self.fc

    def make_dilated(self, *args, **kwargs):
        raise ValueError(
            "Xception encoder does not support dilated mode " "due to pooling operation for downsampling!"
        )

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu, self.conv2, self.bn2, self.relu),
            self.block1,
            self.block2,
            nn.Sequential(
                self.block3,
                self.block4,
                self.block5,
                self.block6,
                self.block7,
                self.block8,
                self.block9,
                self.block10,
                self.block11,
            ),
            nn.Sequential(self.block12, self.conv3, self.bn3, self.relu, self.conv4, self.bn4),
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict):
        # remove linear
        state_dict.pop("fc.bias", None)
        state_dict.pop("fc.weight", None)

        super().load_state_dict(state_dict)


xception_encoders = {
    "xception": {
        "encoder": XceptionEncoder,
        "pretrained_settings": pretrained_settings["xception"],
        "params": {
            "out_channels": (3, 64, 128, 256, 728, 2048),
        },
    },
}
