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
from google.cloud.storage.client import Client
from google.cloud.storage.bucket import Bucket, Blob
from google.cloud.exceptions import GoogleCloudError
from data_connector.gcp.uploader import Uploader

@pytest.mark.parametrize("bucket_name, file_path, blob_name", \
                        [("test-bucket","test_unittest.txt",0),(0,"test_unittest.txt","MtW Test"), \
                        ("test-bucket",0,"MtW Test")])
def test_upload_typeerror(get_fake_storage_client, bucket_name, file_path, blob_name):
    with pytest.raises(TypeError):
        uploader = Uploader(get_fake_storage_client)
        uploader.upload_to_bucket(bucket_name, file_path, blob_name)


def test_upload_Fail(get_fake_storage_client, mocker):

    with pytest.raises(GoogleCloudError):
        bucket_name = "dataconnector_data_bucket"
        file_path = "test_unittest.txt"
        blob_name = "MtW Test"
        mocker.patch("os.path.isfile", return_value=True)
        uploader = Uploader(get_fake_storage_client)
        uploader.upload_to_bucket(bucket_name, file_path, blob_name)


def test_upload_True(get_fake_storage_client, mocker):

    bucket_name = "dataconnector_data_bucket"
    file_path = "test_unittest.txt"
    blob_name = "MtW Test"
    mocker.patch("os.path.isfile", return_value=True)
    mocker.patch.object(Client, "get_bucket", side_effect=Bucket)
    mocker.patch.object(Blob, "upload_from_filename", return_value=True)

    uploader = Uploader(get_fake_storage_client)
    uploader.upload_to_bucket(bucket_name, file_path, blob_name)


def test_upload_filenotfound(get_fake_storage_client):

    bucket_name = "dataconnector_data_bucket"
    file_path = "test_unittest.txt"
    blob_name = "MtW Test"
    with pytest.raises(FileNotFoundError):
        uploader = Uploader(get_fake_storage_client)
        uploader.upload_to_bucket(bucket_name, file_path, blob_name)