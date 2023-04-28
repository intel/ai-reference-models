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


class IntUploader(metaclass=ABCMeta):
    def __init__(self, connector: object):
        self.connector = connector
    
    @abstractmethod
    def upload(self, source_path:str ,container_obj:object):
        pass
    @abstractmethod 
    def upload(self, container_obj: object, data_file: object, object_name: object):
        pass