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
# Azure
from azure.storage.blob import BlobServiceClient, ContainerClient
from azureml.core.model import Model
from azureml.core.dataset import Dataset
from azure.ai.ml import MLClient
from azure.storage.blob import BlobClient, ContainerClient
CONNECTOR_TYPE_ERROR = "Connector for blobs should be a BlobServiceClient"


class Downloader(IntDownloader):
    def __init__(self, connector: object): # BlobServiceClient | MLClient
        if isinstance(connector, (BlobServiceClient, MLClient)):
            super().__init__(connector)
        self.container_client = None

    def download(
        self,
        donwload_obj: object, # ContainerClient | Model | Dataset
        data_file: str,
        destiny: str,
        version: str = None,
    ) -> object: # ContainerClient | BlobClient:
        if isinstance(self.connector, BlobServiceClient):
            blob_service_client = self.connector
            blob_container = donwload_obj
            blob_container_client = blob_service_client.get_container_client(
                blob_container
            )
            storage_stream_downloader = blob_container_client.download_blob(
                data_file
            ).readall()
            with open(destiny, mode="wb") as downloaded_blob:
                downloaded_blob.write(storage_stream_downloader)
            self.container_client = blob_container_client
            return blob_container_client
        if isinstance(donwload_obj, Model):
            if not destiny:
                destiny = "."
            donwload_obj.download(target_dir=destiny, exist_ok=True)
        if isinstance(donwload_obj, Dataset):
            if not version:
                version = "latest"
            donwload_obj.get_by_name(name=data_file, version=version)

    def list_blobs(self, container_obj: object = None)-> iter: # str | ContainerClient | BlobServiceClient = None):
        if isinstance(container_obj, (ContainerClient,str)):
            blob_service_client = self.connector
            container_client = blob_service_client.get_container_client(container_obj)
            container_list = container_client.list_blob_names()
            self.container_client = container_client
            print("Container Client Content")
            for container in container_list:
                print(container)
            return container_list
        elif isinstance(self.connector, BlobServiceClient) and not container_obj:
            print("Containers...")
            containers = self.connector.list_containers()
            for container in containers:
                print(container["name"])
            return containers
        