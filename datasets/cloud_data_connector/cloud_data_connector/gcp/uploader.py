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
import os
from google.cloud.exceptions import GoogleCloudError
class Uploader():

    def __init__(self, connector: object):
        self.connector = connector

    def upload_to_bucket(self, container_obj:object, data_file:object, destination:object) -> object:
        if not isinstance(container_obj, str):
            raise TypeError("container_obj must be str.")
        elif not isinstance(data_file, str):
            raise TypeError("data_file must be str.")
        elif not isinstance(destination, str):
            raise TypeError("destination must be str.")
        
        if not os.path.isfile(data_file):
            raise FileNotFoundError("{} not found.".format(data_file))
        
        try:
            storage_client = self.connector
            bucket = storage_client.get_bucket(container_obj)
            blob = bucket.blob(destination)
            blob.upload_from_filename(data_file)
            return True
        except GoogleCloudError as e:
            print(e)
            raise GoogleCloudError('Failed to copy local file {0} to cloud storage file {1}.'.format(data_file, destination))
