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
from azure.identity import ClientSecretCredential
from azure.core.credentials import AzureNamedKeyCredential
from azure.storage.blob import BlobServiceClient
from azure.ai.ml import MLClient


class Account_Credential:
    def __init__(
        self, url: str, credential: object#  ClientSecretCredential | AzureNamedKeyCredential
    ):
        self.url = url
        self.credential = credential

    def __init__(
        self,
        url: str,
        active_directory_tenant_id: str,
        active_directory_application_id: str,
        active_directory_application_secret: str,
    ):
        self.url = url
        self.credential = ClientSecretCredential(
            active_directory_tenant_id,
            active_directory_application_id,
            active_directory_application_secret,
        )

    def get_url(self):
        return self.url

    def get_credential(self):
        return self.credential

    def create_credential(self, object_type: object): #BlobServiceClient | MLClient):
        pass

    def get_mlclient(self, config_file: MLClient):
        pass

    def set_url(self, url: str):
        self.url = url
