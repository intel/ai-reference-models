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
import dotenv
import os 
from pathlib import Path
# Data Types
from azure.storage.blob import BlobServiceClient

# Data connector
from data_connector.azure import Connector, connect
from data_connector.azure import Uploader, Downloader
# Get the connection string from an environment variable, 
# preferible under file called .env
dotenv.load_dotenv()

# This sample requires a .env file with a variable CONNECTION_STRING 
blob_connection_string = os.getenv('CONNECTION_STRING')

c_path = Path.cwd()
sample_file = f'{c_path}\\sample_data\\credit_card_clients.xls'
# This connector is created from connection string
connector: BlobServiceClient = connect(connection_string=blob_connection_string)
# uploader is the object created to upload files into containers
uploader: Uploader = Uploader(connector=connector)
# Downloader is an object created to download blobs using connector
downloader: Downloader = Downloader(connector=connector)

# This shows all your containers
downloader.list_blobs()
print("_" * 10)
# This method upload test file into a blob container, use your contanier name 
# in blob_container_name
container_name = "data-connector"
uploader.upload(
    source_path= sample_file, 
    blob_container_name=container_name
)
# Download a file first time requires the name of container,
# after this container is set permanently on downloader instance
# Also this method returns a blob container client
_ = downloader.list_blobs(container_obj=container_name)
code_script = f'{c_path}\\src\\main.py'

_ = uploader.upload(
    source_path=code_script, 
    blob_container_name=container_name
    )


# Now archives are on the blob container
print("_" * 10)
downloader.list_blobs()