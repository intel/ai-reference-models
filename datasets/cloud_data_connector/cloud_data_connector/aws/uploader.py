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
from cloud_data_connector.int.int_uploader import IntUploader
from botocore.exceptions import ClientError
import logging
import botocore

class Uploader(IntUploader):
    """
    This is a class for upload operations for AWS S3 buckets.

    Attributes:
        connector (botocore.client.BaseClient): An AWS S3 BaseClient
    """

    def __init__(self, connector: botocore.client.BaseClient):
        """
        The constructor for AWS S3 Uploader.

        Args:
            connector (botocore.client.BaseClient): An AWS S3 BaseClient.
        """
        super().__init__(connector)
    
    def upload(self, container_obj: str, data_file: str, object_name: str):
        """
        The function to upload a file to an S3 bucket.

        Args:
            container_obj (str): The name of the bucket to upload to.
            data_file (str): The path to the file to upload.
            object_name (str): The name of the file to upload to.
        
        Raises:
            Exception: Client Error message.
        """
        # If S3 object_name was not specified, use file_name
        if object_name is None:
            object_name = os.path.basename(data_file)

        s3_client = self.connector
        try:
            response = s3_client.upload_file(data_file, container_obj, object_name)
        except ClientError as error:
            code = error.response['Error']['Code']
            message = error.response['Error']['Message']
            logging.error(error)
            raise Exception(f"Connection object error. Code: {code}. Message: {message}")