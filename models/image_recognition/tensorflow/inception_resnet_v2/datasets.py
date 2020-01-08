#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Intel Corporation
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


# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Benchmark dataset utilities.
"""

import os
import tensorflow as tf
from abc import abstractmethod

import preprocessing

IMAGENET_NUM_TRAIN_IMAGES = 1281167
IMAGENET_NUM_VAL_IMAGES = 50000


def create_dataset(data_dir, data_name):
    """Create a Dataset instance based on data_dir and data_name."""

    supported_datasets = {
        'imagenet': ImagenetData,
    }

    if not data_dir and not data_name:
        # When using synthetic data, use synthetic imagenet images by default.
        data_name = 'imagenet'

    if data_name is None:
        for supported_name in supported_datasets:
            if supported_name in data_dir:
                data_name = supported_name
                break

    if data_name is None:
        raise ValueError('Could not identify name of dataset. '
                         'Please specify with --data_name option.')

    if data_name not in supported_datasets:
        raise ValueError('Unknown dataset. Must be one of %s', ', '.join(
            [key for key in sorted(supported_datasets.keys())]))

    return supported_datasets[data_name](data_dir)


class Dataset(object):
    """Abstract class for cnn benchmarks dataset."""

    def __init__(self, name, height=None, width=None, depth=None, data_dir=None,
                 queue_runner_required=False, num_classes=1000):
        self.name = name
        self.height = height
        self.width = width
        self.depth = depth or 3

        self.data_dir = data_dir
        self._queue_runner_required = queue_runner_required
        self._num_classes = num_classes

    def tf_record_pattern(self, subset):
        return os.path.join(self.data_dir, '%s-*-of-*' % subset)

    def reader(self):
        return tf.TFRecordReader()

    @property
    def num_classes(self):
        return self._num_classes

    @num_classes.setter
    def num_classes(self, val):
        self._num_classes = val

    @abstractmethod
    def num_examples_per_epoch(self, subset):
        pass

    def __str__(self):
        return self.name

    def get_image_preprocessor(self):
        return None

    def queue_runner_required(self):
        return self._queue_runner_required

    def use_synthetic_gpu_images(self):
        return not self.data_dir


class ImagenetData(Dataset):
    """Configuration for Imagenet dataset."""

    def __init__(self, data_dir=None):
        super(ImagenetData, self).__init__('imagenet', 300, 300, data_dir=data_dir)

    def num_examples_per_epoch(self, subset='train'):
        if subset == 'train':
            return IMAGENET_NUM_TRAIN_IMAGES
        elif subset == 'validation':
            return IMAGENET_NUM_VAL_IMAGES
        else:
            raise ValueError('Invalid data subset "%s"' % subset)

    def get_image_preprocessor(self):
        if self.use_synthetic_gpu_images():
            return preprocessing.SyntheticImagePreprocessor
        else:
            return preprocessing.RecordInputImagePreprocessor
