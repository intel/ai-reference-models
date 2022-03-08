#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
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

# Dictionary of TFHub models
tfhub_model_map = {
    "resnet_v1_50": {
        "imagenet_model": "https://tfhub.dev/google/imagenet/resnet_v1_50/classification/5",
        "feature_vector": "https://tfhub.dev/google/imagenet/resnet_v1_50/feature_vector/5",
        "image_size": 224
    },
    "resnet_v2_50": {
        "imagenet_model": "https://tfhub.dev/google/imagenet/resnet_v2_50/classification/5",
        "feature_vector": "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5",
        "image_size": 224
    },
    "resnet_v2_101": {
        "imagenet_model": "https://tfhub.dev/google/imagenet/resnet_v2_101/classification/5",
        "feature_vector": "https://tfhub.dev/google/imagenet/resnet_v2_101/feature_vector/5",
        "image_size": 224
    },
    "mobilenet_v2_100_224": {
        "imagenet_model": "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/5",
        "feature_vector": "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4",
        "image_size": 224
    },
    "efficientnetv2-s": {
        "imagenet_model": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_s/classification/2",
        "feature_vector": "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_s/feature_vector/2",
        "image_size": 384
    },
    "efficientnet_b0": {
        "imagenet_model": "https://tfhub.dev/google/efficientnet/b0/classification/1",
        "feature_vector": "https://tfhub.dev/google/efficientnet/b0/feature-vector/1",
        "image_size": 224
    },
    "efficientnet_b1": {
        "imagenet_model": "https://tfhub.dev/google/efficientnet/b1/classification/1",
        "feature_vector": "https://tfhub.dev/google/efficientnet/b1/feature-vector/1",
        "image_size": 240
    },
    "efficientnet_b2": {
        "imagenet_model": "https://tfhub.dev/google/efficientnet/b2/classification/1",
        "feature_vector": "https://tfhub.dev/google/efficientnet/b2/feature-vector/1",
        "image_size": 260
    },
    "inception_v3": {
        "imagenet_model": "https://tfhub.dev/google/imagenet/inception_v3/classification/5",
        "feature_vector": "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/5",
        "image_size": 299
    },
    "nasnet_large": {
        "imagenet_model": "https://tfhub.dev/google/imagenet/nasnet_large/classification/5",
        "feature_vector": "https://tfhub.dev/google/imagenet/nasnet_large/feature_vector/5",
        "image_size": 331
    }
}

