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
from pathlib import Path
# azure
from azure.storage.blob import BlobServiceClient, ContainerClient
from azureml.core.environment import DockerBuildContext, Environment
from azureml.core.workspace import Workspace
from azure.ai.ml import MLClient
# Exceptions
from azure.core.exceptions import ResourceExistsError
# Interface
from cloud_data_connector.int.int_uploader import IntUploader


DOCKER_FILE_NOT_EXISTS = "Docker file not exists "
FILE_SOURCE_NOT_FOUND = "Source file not found "
BAD_CONNECTOR_INSTANCE = "Connector instance is not correct "
BLOB_ALREADY_EXISTS = "Blob file already exist with this name"

class Uploader(IntUploader):
    def __init__(self, connector: object)-> None:#  BlobServiceClient | MLClient
        if isinstance(connector, (BlobServiceClient, MLClient)):
            self.connector = connector
        else:
            raise ValueError(BAD_CONNECTOR_INSTANCE + f"{type(connector)}")
        self.container_client = None

    def upload(self, source_path: str, blob_container_name: str):
        """Upload data to azure
        Keyword arguments:
        source_path -- the source path of file to upload
        blob_container_name -- name of container where file will be loaded
        """
        
        if isinstance(self.container_client, ContainerClient):
            if os.path.exists(source_path):
                name = Path(source_path).name
                with open(source_path, 'rb') as data: 
                    self.container_client.upload_blob(
                        name=name,
                        data=data
                    )
                    print(f"File {source_path} pushed")
        if isinstance(self.connector, BlobServiceClient):
            container_client = self.connector.get_container_client(blob_container_name)
            if os.path.exists(source_path):
                try:
                    with open(source_path, 'rb') as data:
                        name = Path(source_path).name
                        container_client.upload_blob(name=name, data=data)
                        print(f"File {source_path} pushed")
                        self.container_client = container_client
                except ResourceExistsError as ree:
                    print(ree)
                    print(BLOB_ALREADY_EXISTS)

            else:
                raise FileNotFoundError(FILE_SOURCE_NOT_FOUND + f" {source_path} ")
        else:
            raise ValueError(BAD_CONNECTOR_INSTANCE + f"{type(self.connector)}")

    def uploadml(
        self, docker_build_file_context_path: str, workspace_name: str, path: str
    ) -> DockerBuildContext:
        try:
            _workspace = self.connector.workspaces.get(workspace_name)
            if os.path.exists(docker_build_file_context_path):
                _ = DockerBuildContext().from_local_directory(
                    workspace=_workspace,
                    path=path,
                    dockerfile_path=docker_build_file_context_path,
                )
                return _
            else:
                raise FileExistsError(
                    DOCKER_FILE_NOT_EXISTS + f" {docker_build_file_context_path}"
                )
        except Exception as ex:
            # Assuming error is on workspace name
            print(ex)
            _ = self.connector.workspaces.list()
            print(_)

    def create_environment_form_docker(
        self,
        name: str,
        docker_file: str,
        conda_specification: str = None,
        pip_requirements: str = None,
    ):
        if os.path.exists(docker_file):
            _ = Environment.from_dockerfile(
                name=name,
                dockerfile=docker_file,
                conda_specification=conda_specification,
                pip_requirements=pip_requirements,
            )
        else:
            raise FileExistsError(DOCKER_FILE_NOT_EXISTS + f" {docker_file}")
