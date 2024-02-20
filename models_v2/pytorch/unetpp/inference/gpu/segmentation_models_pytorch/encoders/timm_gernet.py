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

from timm.models import ByoModelCfg, ByoBlockCfg, ByobNet

from ._base import EncoderMixin
import torch.nn as nn


class GERNetEncoder(ByobNet, EncoderMixin):
    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3

        del self.head

    def get_stages(self):
        return [
            nn.Identity(),
            self.stem,
            self.stages[0],
            self.stages[1],
            self.stages[2],
            nn.Sequential(self.stages[3], self.stages[4], self.final_conv),
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("head.fc.weight", None)
        state_dict.pop("head.fc.bias", None)
        super().load_state_dict(state_dict, **kwargs)


regnet_weights = {
    "timm-gernet_s": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-ger-weights/gernet_s-756b4751.pth",  # noqa
    },
    "timm-gernet_m": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-ger-weights/gernet_m-0873c53a.pth",  # noqa
    },
    "timm-gernet_l": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-ger-weights/gernet_l-f31e2e8d.pth",  # noqa
    },
}

pretrained_settings = {}
for model_name, sources in regnet_weights.items():
    pretrained_settings[model_name] = {}
    for source_name, source_url in sources.items():
        pretrained_settings[model_name][source_name] = {
            "url": source_url,
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000,
        }

timm_gernet_encoders = {
    "timm-gernet_s": {
        "encoder": GERNetEncoder,
        "pretrained_settings": pretrained_settings["timm-gernet_s"],
        "params": {
            "out_channels": (3, 13, 48, 48, 384, 1920),
            "cfg": ByoModelCfg(
                blocks=(
                    ByoBlockCfg(type="basic", d=1, c=48, s=2, gs=0, br=1.0),
                    ByoBlockCfg(type="basic", d=3, c=48, s=2, gs=0, br=1.0),
                    ByoBlockCfg(type="bottle", d=7, c=384, s=2, gs=0, br=1 / 4),
                    ByoBlockCfg(type="bottle", d=2, c=560, s=2, gs=1, br=3.0),
                    ByoBlockCfg(type="bottle", d=1, c=256, s=1, gs=1, br=3.0),
                ),
                stem_chs=13,
                stem_pool=None,
                num_features=1920,
            ),
        },
    },
    "timm-gernet_m": {
        "encoder": GERNetEncoder,
        "pretrained_settings": pretrained_settings["timm-gernet_m"],
        "params": {
            "out_channels": (3, 32, 128, 192, 640, 2560),
            "cfg": ByoModelCfg(
                blocks=(
                    ByoBlockCfg(type="basic", d=1, c=128, s=2, gs=0, br=1.0),
                    ByoBlockCfg(type="basic", d=2, c=192, s=2, gs=0, br=1.0),
                    ByoBlockCfg(type="bottle", d=6, c=640, s=2, gs=0, br=1 / 4),
                    ByoBlockCfg(type="bottle", d=4, c=640, s=2, gs=1, br=3.0),
                    ByoBlockCfg(type="bottle", d=1, c=640, s=1, gs=1, br=3.0),
                ),
                stem_chs=32,
                stem_pool=None,
                num_features=2560,
            ),
        },
    },
    "timm-gernet_l": {
        "encoder": GERNetEncoder,
        "pretrained_settings": pretrained_settings["timm-gernet_l"],
        "params": {
            "out_channels": (3, 32, 128, 192, 640, 2560),
            "cfg": ByoModelCfg(
                blocks=(
                    ByoBlockCfg(type="basic", d=1, c=128, s=2, gs=0, br=1.0),
                    ByoBlockCfg(type="basic", d=2, c=192, s=2, gs=0, br=1.0),
                    ByoBlockCfg(type="bottle", d=6, c=640, s=2, gs=0, br=1 / 4),
                    ByoBlockCfg(type="bottle", d=5, c=640, s=2, gs=1, br=3.0),
                    ByoBlockCfg(type="bottle", d=4, c=640, s=1, gs=1, br=3.0),
                ),
                stem_chs=32,
                stem_pool=None,
                num_features=2560,
            ),
        },
    },
}
