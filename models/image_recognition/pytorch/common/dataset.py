#
# Copyright (c) 2021 Intel Corporation
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

import os
import utils
import presets

import torch
import torchvision
import torch.utils.data
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as transforms

def _get_cache_path(filepath):
    import hashlib
    h = hashlib.sha224(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path

def get_transform_params(model):
    resize_size, crop_size = 256, 224
    interpolation = InterpolationMode.BILINEAR
    if model == "inception_v3":
        resize_size, crop_size = 342, 299
    elif model.startswith("efficientnet_"):
        sizes = {
            "b0": (256, 224),
            "b1": (256, 240),
            "b2": (288, 288),
            "b3": (320, 300),
            "b4": (384, 380),
            "b5": (456, 456),
            "b6": (528, 528),
            "b7": (600, 600),
        }
        e_type = model.replace("efficientnet_", "")
        resize_size, crop_size = sizes[e_type]
        interpolation = InterpolationMode.BICUBIC
    return resize_size, crop_size, interpolation

def get_imagenet_test_data_loader(args):
    valdir = os.path.join(args.data_path, "val")
    resize_size, crop_size, interpolation = get_transform_params(args.arch)
    cache_path = _get_cache_path(valdir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print("Loading dataset_test from {}".format(cache_path))
        dataset_test, _ = torch.load(cache_path)
    else:
        dataset_test = torchvision.datasets.ImageFolder(
            valdir,
            presets.ClassificationPresetEval(crop_size=crop_size, resize_size=resize_size, interpolation=interpolation)
        )
        if args.cache_dataset:
            print("Saving dataset_test to {}".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

    print("Creating data loaders")
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    
    imagenet_test_data_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True
    )
    return imagenet_test_data_loader

