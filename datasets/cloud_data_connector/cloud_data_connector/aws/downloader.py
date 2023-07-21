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
import botocore.exceptions
import botocore.client
import logging

# Set up our logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

class Downloader(IntDownloader):
    """
    This is a class for download operations for AWS S3 buckets.
      
    Attributes:
        connector (botocore.client.BaseClient): An AWS S3 BaseClient
    """
    
    def __init__(self, connector: botocore.client.BaseClient):
        """
        The constructor for AWS S3 Downloader.

        Args:
            connector (botocore.client.BaseClient): An AWS S3 BaseClient.
        """
        super().__init__(connector)

    def download(self, container_obj:str, data_file:str, destiny:str):
        """
        The function to download a file from a S3 bucket.

        Args:
            container_obj (str): The name of the bucket to download from.
            data_file (str): The name of the file to download from.
            destiny (str): The path to the file to download to.

        """
        if isinstance(self.connector, botocore.client.BaseClient):
            s3_client = self.connector
            s3_client.download_file(container_obj, data_file, destiny)

    def list_blobs(self, container_obj:str) -> list:
        """
        The function to get a list of the objects in a bucket.

        Args:
            container_obj (str): The bucket name to list.

        Raises:
            Exception: Client Error message.
            Exception: Boto Core Error message. 

        Returns:
            list: A list of object names contained in the bucket.
        """
        container_list = []
        s3_client = self.connector
        try:
            response = s3_client.list_objects_v2(Bucket=container_obj)
            for file in response['Contents']:
                container_list.append(file['Key'])
            return container_list
        except botocore.exceptions.ClientError as error:
            code = error.response['Error']['Code']
            message = error.response['Error']['Message']
            logging.error(error)
            raise Exception(f"Connection object error. Code: {code}. Message: {message}")
        except botocore.exceptions.BotoCoreError as error:
            logging.error(error)
            raise Exception(error.fmt)
        