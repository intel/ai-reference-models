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
import boto3
import botocore
from cloud_data_connector.int.int_connector import IntConnector

class Connector(IntConnector):
    """
    This is a class for create a connection to AWS S3 buckets.

    """
    def __init__(self):
        super().__init__()

    def connect(self, connection_string: str = "", connector_object: object = None) -> botocore.client.BaseClient:
        if connection_string == "" and connector_object == None:
            return self.connect_aws()
        else:
            raise TypeError("Connection type not supported for AWS S3")
    
    def connect_aws(self) -> botocore.client.BaseClient:
        """
        The function to create a connection to AWS S3 buckets.
        It creates a boto3.Session using the access keys defined by
        default configurations for AWS.

        Returns:
            botocore.client.BaseClient: A S3 client.
        """
        aws_session = boto3.Session()
        s3_client = aws_session.client('s3')
        return s3_client