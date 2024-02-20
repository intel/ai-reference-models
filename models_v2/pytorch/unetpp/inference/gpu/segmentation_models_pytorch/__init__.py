#
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
# SPDX-License-Identifier: EPL-2.0
#

from .decoders.unetplusplus import UnetPlusPlus

from .__version__ import __version__

# some private imports for create_model function
from typing import Optional as _Optional
import torch as _torch


def create_model(
    arch: str,
    encoder_name: str = "resnet34",
    encoder_weights: _Optional[str] = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
    **kwargs,
) -> _torch.nn.Module:
    """Models entrypoint, allows to create any model architecture just with
    parameters, without using its class
    """

    archs = [
        UnetPlusPlus,
    ]
    archs_dict = {a.__name__.lower(): a for a in archs}
    try:
        model_class = archs_dict[arch.lower()]
    except KeyError:
        raise KeyError(
            "Wrong architecture type `{}`. Available options are: {}".format(
                arch,
                list(archs_dict.keys()),
            )
        )
    return model_class(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        **kwargs,
    )
