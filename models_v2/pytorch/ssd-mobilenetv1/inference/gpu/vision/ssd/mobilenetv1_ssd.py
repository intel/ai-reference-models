# Copyright (c) 2023 Intel Corporation
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
#
# MIT License
#
# Copyright (c) 2019 Hao Gao
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
#

import torch
from torch.nn import Conv2d, Sequential, ModuleList, ReLU
from ..nn.mobilenet import MobileNetV1

from .ssd import SSD
from .predictor import Predictor
from .config import mobilenetv1_ssd_config as config


def create_mobilenetv1_ssd(num_classes, is_test=False, device=None):
    base_net = MobileNetV1(1001).model  # disable dropout layer

    source_layer_indexes = [
        12,
        14,
    ]
    extras = ModuleList(
        [
            Sequential(
                Conv2d(in_channels=1024, out_channels=256, kernel_size=1),
                ReLU(),
                Conv2d(
                    in_channels=256,
                    out_channels=512,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                ReLU(),
            ),
            Sequential(
                Conv2d(in_channels=512, out_channels=128, kernel_size=1),
                ReLU(),
                Conv2d(
                    in_channels=128,
                    out_channels=256,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                ReLU(),
            ),
            Sequential(
                Conv2d(in_channels=256, out_channels=128, kernel_size=1),
                ReLU(),
                Conv2d(
                    in_channels=128,
                    out_channels=256,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                ReLU(),
            ),
            Sequential(
                Conv2d(in_channels=256, out_channels=128, kernel_size=1),
                ReLU(),
                Conv2d(
                    in_channels=128,
                    out_channels=256,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                ReLU(),
            ),
        ]
    )

    regression_headers = ModuleList(
        [
            Conv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1),
            Conv2d(in_channels=1024, out_channels=6 * 4, kernel_size=3, padding=1),
            Conv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1),
            Conv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
            Conv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
            Conv2d(
                in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1
            ),  # TODO: change to kernel_size=1, padding=0?
        ]
    )

    classification_headers = ModuleList(
        [
            Conv2d(
                in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1
            ),
            Conv2d(
                in_channels=1024, out_channels=6 * num_classes, kernel_size=3, padding=1
            ),
            Conv2d(
                in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1
            ),
            Conv2d(
                in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1
            ),
            Conv2d(
                in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1
            ),
            Conv2d(
                in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1
            ),  # TODO: change to kernel_size=1, padding=0?
        ]
    )

    return SSD(
        num_classes,
        base_net,
        source_layer_indexes,
        extras,
        classification_headers,
        regression_headers,
        is_test=is_test,
        config=config,
        device=device,
    )


def create_mobilenetv1_ssd_predictor(
    net, candidate_size=200, nms_method=None, sigma=0.5, device=None
):
    predictor = Predictor(
        net,
        config.image_size,
        config.image_mean,
        config.image_std,
        nms_method=nms_method,
        iou_threshold=config.iou_threshold,
        candidate_size=candidate_size,
        sigma=sigma,
        device=device,
    )
    return predictor
