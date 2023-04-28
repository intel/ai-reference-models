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
import pytest
import os
import boto3
import botocore
from tempfile import NamedTemporaryFile
from data_connector.aws.connector import Connector
from data_connector.aws.downloader import Downloader
from data_connector.aws.uploader import Uploader

def test_connect(aws_credentials):
    # testing the Connector class and create a s3_client
    connector = Connector()
    s3_client = connector.connect()
    assert isinstance(s3_client, botocore.client.BaseClient), "S3 client is not created"

def test_connect_with_connection_string(aws_credentials):
    # testing the Connector class and create a s3_client
    """Check TypeError raised when passed a connection string"""
    with pytest.raises(TypeError):
        connector = Connector()
        resource = boto3.resource('s3')
        connector.connect(connection_string="conection_string")

def test_connect_error(aws_credentials):
    """Check TypeError raised when passed an object not compatible"""
    with pytest.raises(TypeError):
        connector = Connector()
        resource = boto3.resource('s3')
        connector.connect(connector_object=resource)

def test_list_blobs(aws_session, s3_test):
    bucket_name = 'my-test-bucket'
    file_text = "test"
    with NamedTemporaryFile(delete=True, suffix=".csv") as tmp:
        with open(tmp.name, "w", encoding="UTF-8") as f:
            f.write(file_text)
        # create a connector using a aws session
        connector = Connector()
        connection_obj = connector.connect()
        # create an Uploader using a connector object
        uploader = Uploader(connection_obj)
        # upload two files for testing
        uploader.upload(bucket_name, tmp.name, "file1.csv")
        uploader.upload(bucket_name, tmp.name, "file2.csv")
        
        # create a Downloader object to list object/files from a bucket
        downloader = Downloader(connection_obj)
        actual = downloader.list_blobs(bucket_name)
        assert actual == ["file1.csv","file2.csv"], "Files listed incorrectly!"
    
def test_download(aws_session, s3_test):
    bucket_name = 'my-test-bucket'
    file_text = "test"
    with NamedTemporaryFile(delete=True, suffix=".csv") as tmp:
        with open(tmp.name, "w", encoding="UTF-8") as f:
            f.write(file_text)
        # testing the Downloader class and download a file
        connector = Connector()
        connector_obj = connector.connect()
        # create an Uploader and a file for testing
        uploader = Uploader(connector_obj)
        uploader.upload(bucket_name, tmp.name, "file3.csv")
        # download file named file3.csv for testing
        downloader = Downloader(connector_obj)
        # download file from bucket and name it example.csv
        downloader.download(bucket_name, "file3.csv", "example.csv")
        # check if file was downloaded and exist in our local dir
        assert os.path.isfile('example.csv'), "File not downloaded"
        # delete files downloaded
        os.remove("example.csv")

def test_upload(aws_session, s3_test):
    bucket_name = 'my-test-bucket'
    file_text = "test_upload"
    with NamedTemporaryFile(delete=True, suffix=".csv") as tmp:
        with open(tmp.name, "w", encoding="UTF-8") as f:
            f.write(file_text)
        # create a Downloader object using a connector object 
        connector = Connector()
        connector_obj = connector.connect()
        uploader = Uploader(connector_obj)
        # upload a file to bucket
        uploader.upload(bucket_name, tmp.name, "file4.csv")
        # list all files from bucket and check if uploaded file exists
        downloader = Downloader(connector_obj)
        list_blobs = downloader.list_blobs(bucket_name)
        assert list_blobs == ["file4.csv"]

if __name__=='__main__':
    pytest.main()