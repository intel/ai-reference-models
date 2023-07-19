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
from cloud_data_connector.int.int_downloader import IntDownloader
from google.api_core.exceptions import NotFound
from google.cloud.storage.client import Client
from abc import ABCMeta

class Downloader(IntDownloader, metaclass=ABCMeta):
    
    def __init__(self, connector: object):
        if not isinstance(connector, Client):
            raise TypeError("connector must be {}.".format(Client))
        
        super().__init__(connector)


    def download(self, container_obj:object, data_file:object, destination:object) -> object:
        if not isinstance(container_obj, str):
            raise TypeError("container_obj must be str.")
        elif not isinstance(data_file, str):
            raise TypeError("data_file must be str.")
        elif not isinstance(destination, str):
            raise TypeError("destination must be str.")
        
        try:
            storage_client = self.connector
            bucket = storage_client.get_bucket(container_obj)
            blob = bucket.blob(data_file)
            with open(destination, 'wb') as f:
                self.connector.download_blob_to_file(blob, f)
            return True
        except NotFound as e:
            print("blob not found: {}".format(e))
            return False


    def list_blobs(self, container_obj:object):
        """Lists all blobs."""

        if not isinstance(container_obj, str):
            raise TypeError("container_obj must be str.")
        
        try:
            storage_client = self.connector
            bucket = storage_client.get_bucket(container_obj)
            blobs = list(bucket.list_blobs())
            print(blobs)
            if blobs != []:
                for blob in blobs:
                    print(blob.name)
            else:
                print("Empty blob list.")
        except NotFound as e:
            print("Bucket not found: {}".format(e))
            return []
        
        return blobs


    def list_buckets(self):
        """Lists all buckets."""
        try:
            storage_client = self.connector
            buckets = list(storage_client.list_buckets())

            for bucket in buckets:
                print(bucket.name)
        except NotFound as e:
            print("Not found: {}".format(e))
            return []
        
        return buckets