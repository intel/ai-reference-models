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
from cloud_data_connector.int.int_connector import IntConnector

# Python
from os.path import exists

# Azure Blob
from azure.storage.blob import BlobServiceClient

# Azure ML
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Workspace

# Azure Identity
from azure.identity import (
    DefaultAzureCredential,
    EnvironmentCredential,
    ManagedIdentityCredential,
    SharedTokenCacheCredential,
    AzureCliCredential
)
from azure.core.credentials import TokenCredential

# BLOB Credentials
# AzureNamedKeyCredential,  AzureSasCredential
# MLService Credentials
# AzureNamedKeyCredential, AzureSasCredential


CONFIGURATION_FILE_MISSED_MSG = "Configuration file to connect Azure service is missed"
CONNECTION_ERROR_MSG = "Connection is not succeed. Verify Connection string and login using 'az login' and review config.json file values"
CONNECTION_AZUREML_NOT_CREATED = "Connection to azure ml is not made..."

DEFAULT_CONF_FILE = "config.json"


class Connector(IntConnector):
    def __init__(self):
        self.connector = None

    def connect(
        self, connection_string: str = "", connector_object: object = None
    ) -> object: # BlobServiceClient | MLClient:
        connection = None
        if connection_string:
            connection: BlobServiceClient = self.connect_by_connection_string(
                connection_string=connection_string
            )
        if not connection_string and not connector_object:
            try:
                connection: MLClient = self.connect_from_config_file()
                return connection
            except FileNotFoundError as fne:
                print(fne)

        if not connection:
            raise ConnectionError(CONNECTION_ERROR_MSG)
        else:
            self.connector = connection
            return self.connector

    def connect_by_connection_string(self, connection_string: str) -> BlobServiceClient:
        blob_service_client: BlobServiceClient = (
            BlobServiceClient.from_connection_string(connection_string)
        )
        return blob_service_client

    def connect_from_config_file(
        self, conf_file: str = DEFAULT_CONF_FILE, credential: DefaultAzureCredential = None
    ) -> MLClient:
        if exists(conf_file):
            if credential:
                workspace_ml_client: Workspace = MLClient.from_config(
                    credential=credential, file_name=conf_file
                )
            else:
                workspace_ml_client: Workspace = MLClient.from_config(
                    AzureCliCredential(), file_name=conf_file
                )
            if not workspace_ml_client:
                raise RuntimeError(CONNECTION_AZUREML_NOT_CREATED)
            return workspace_ml_client
        else:
            raise FileNotFoundError(CONFIGURATION_FILE_MISSED_MSG)

    def get_connector(self) -> object: #BlobServiceClient | MLClient
        if self.connector:
            return self.connector
        else:
            raise ConnectionError(CONNECTION_ERROR_MSG)
