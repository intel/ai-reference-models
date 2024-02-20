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

import timm
import numpy as np
import torch.nn as nn

from ._base import EncoderMixin


def _make_divisible(x, divisible_by=8):
    return int(np.ceil(x * 1.0 / divisible_by) * divisible_by)


class MobileNetV3Encoder(nn.Module, EncoderMixin):
    def __init__(self, model_name, width_mult, depth=5, **kwargs):
        super().__init__()
        if "large" not in model_name and "small" not in model_name:
            raise ValueError("MobileNetV3 wrong model name {}".format(model_name))

        self._mode = "small" if "small" in model_name else "large"
        self._depth = depth
        self._out_channels = self._get_channels(self._mode, width_mult)
        self._in_channels = 3

        # minimal models replace hardswish with relu
        self.model = timm.create_model(
            model_name=model_name,
            scriptable=True,  # torch.jit scriptable
            exportable=True,  # onnx export
            features_only=True,
        )

    def _get_channels(self, mode, width_mult):
        if mode == "small":
            channels = [16, 16, 24, 48, 576]
        else:
            channels = [16, 24, 40, 112, 960]
        channels = [
            3,
        ] + [_make_divisible(x * width_mult) for x in channels]
        return tuple(channels)

    def get_stages(self):
        if self._mode == "small":
            return [
                nn.Identity(),
                nn.Sequential(
                    self.model.conv_stem,
                    self.model.bn1,
                    self.model.act1,
                ),
                self.model.blocks[0],
                self.model.blocks[1],
                self.model.blocks[2:4],
                self.model.blocks[4:],
            ]
        elif self._mode == "large":
            return [
                nn.Identity(),
                nn.Sequential(
                    self.model.conv_stem,
                    self.model.bn1,
                    self.model.act1,
                    self.model.blocks[0],
                ),
                self.model.blocks[1],
                self.model.blocks[2],
                self.model.blocks[3:5],
                self.model.blocks[5:],
            ]
        else:
            ValueError("MobileNetV3 mode should be small or large, got {}".format(self._mode))

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("conv_head.weight", None)
        state_dict.pop("conv_head.bias", None)
        state_dict.pop("classifier.weight", None)
        state_dict.pop("classifier.bias", None)
        self.model.load_state_dict(state_dict, **kwargs)


mobilenetv3_weights = {
    "tf_mobilenetv3_large_075": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_large_075-150ee8b0.pth"  # noqa
    },
    "tf_mobilenetv3_large_100": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_large_100-427764d5.pth"  # noqa
    },
    "tf_mobilenetv3_large_minimal_100": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_large_minimal_100-8596ae28.pth"  # noqa
    },
    "tf_mobilenetv3_small_075": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_small_075-da427f52.pth"  # noqa
    },
    "tf_mobilenetv3_small_100": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_small_100-37f49e2b.pth"  # noqa
    },
    "tf_mobilenetv3_small_minimal_100": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_small_minimal_100-922a7843.pth"  # noqa
    },
}

pretrained_settings = {}
for model_name, sources in mobilenetv3_weights.items():
    pretrained_settings[model_name] = {}
    for source_name, source_url in sources.items():
        pretrained_settings[model_name][source_name] = {
            "url": source_url,
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "input_space": "RGB",
        }


timm_mobilenetv3_encoders = {
    "timm-mobilenetv3_large_075": {
        "encoder": MobileNetV3Encoder,
        "pretrained_settings": pretrained_settings["tf_mobilenetv3_large_075"],
        "params": {"model_name": "tf_mobilenetv3_large_075", "width_mult": 0.75},
    },
    "timm-mobilenetv3_large_100": {
        "encoder": MobileNetV3Encoder,
        "pretrained_settings": pretrained_settings["tf_mobilenetv3_large_100"],
        "params": {"model_name": "tf_mobilenetv3_large_100", "width_mult": 1.0},
    },
    "timm-mobilenetv3_large_minimal_100": {
        "encoder": MobileNetV3Encoder,
        "pretrained_settings": pretrained_settings["tf_mobilenetv3_large_minimal_100"],
        "params": {"model_name": "tf_mobilenetv3_large_minimal_100", "width_mult": 1.0},
    },
    "timm-mobilenetv3_small_075": {
        "encoder": MobileNetV3Encoder,
        "pretrained_settings": pretrained_settings["tf_mobilenetv3_small_075"],
        "params": {"model_name": "tf_mobilenetv3_small_075", "width_mult": 0.75},
    },
    "timm-mobilenetv3_small_100": {
        "encoder": MobileNetV3Encoder,
        "pretrained_settings": pretrained_settings["tf_mobilenetv3_small_100"],
        "params": {"model_name": "tf_mobilenetv3_small_100", "width_mult": 1.0},
    },
    "timm-mobilenetv3_small_minimal_100": {
        "encoder": MobileNetV3Encoder,
        "pretrained_settings": pretrained_settings["tf_mobilenetv3_small_minimal_100"],
        "params": {"model_name": "tf_mobilenetv3_small_minimal_100", "width_mult": 1.0},
    },
}
