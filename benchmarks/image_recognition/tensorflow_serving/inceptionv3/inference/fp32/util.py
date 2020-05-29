#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Intel Corporation
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

from __future__ import print_function

import tensorflow.compat.v1 as tf


def preprocess_image(image_buffer, model, image_size):
    """Preprocess JPEG encoded bytes to 3D float Tensor."""

    # Decode the string as an RGB JPEG of unknown height and width.
    image = tf.image.decode_jpeg(image_buffer, channels=3)
    # Convert pixels to [0, 1)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # Crop the central region to 87.5% of the original image.
    image = tf.image.central_crop(image, central_fraction=0.875)
    # Resize the image to image_size x image_size.
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [image_size, image_size], align_corners=False)
    image = tf.squeeze(image, [0])
    if model == 'resnet50':
        # For ResNet50, rescale to [0, 256]
        image = tf.multiply(image, 256.0)
    elif model == 'Inceptionv3':
        # For InceptionV3, rescale to [-1, 1]
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
    return image


def parse_example_proto(example_serialized):
    # Dense features in Example proto.
    feature_map = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                            default_value=''),
        'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                                default_value=-1),
    }

    features = tf.parse_single_example(example_serialized, feature_map)
    label = tf.cast(features['image/class/label'], dtype=tf.int32)

    return features['image/encoded'], label
