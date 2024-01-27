#
# -*- coding: utf-8 -*-
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
from abc import ABCMeta, abstractmethod 

#

<<<<<<<< HEAD:benchmarks/recommendation/tensorflow/dien/training/bfloat16/model_init.py
from recommendation.tensorflow.dien.training.dien_model_init import DienModelInitializer


class ModelInitializer(DienModelInitializer):
    """Model initializer for DIEN bf16 Training"""

    def __init__(self, args, custom_args=[], platform_util=None):
        super(ModelInitializer, self).__init__(args, custom_args, platform_util)
========
class IntDownloader(metaclass=ABCMeta):
    
    def __init__(self, connector: object):
            self.connector = connector

    @abstractmethod 
    def download(self, container_obj: object, data_file: object, destiny: object) -> object:
        pass
>>>>>>>> r3.1:datasets/cloud_data_connector/cloud_data_connector/int/int_downloader.py
