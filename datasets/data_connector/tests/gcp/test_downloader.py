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
import tempfile
from google.cloud.storage.blob import Blob
from google.cloud.storage.bucket import Bucket
from data_connector.gcp.downloader import Downloader

def test_download_init():
    try:
        Downloader("")
    except TypeError:
        assert True
    else:
        assert False

def test_download_notfound(get_fake_storage_client):
    f = tempfile.NamedTemporaryFile(delete=False)
    f.close()

    downloader = Downloader(get_fake_storage_client)
    bucket_name = "test-bucket"
    file_path = f.name
    blob_name = "MtW Test"
    assert downloader.download(bucket_name,blob_name,file_path) == False

@pytest.mark.parametrize("bucket_name, file_path, blob_name", \
                        [("test-bucket","test_unittest.txt",0),(0,"test_unittest.txt","MtW Test"), \
                        ("test-bucket",0,"MtW Test")])
def test_download_typeerror(get_fake_storage_client, bucket_name, file_path, blob_name):
    with pytest.raises(TypeError):
        downloader = Downloader(get_fake_storage_client)
        downloader.download(bucket_name,blob_name,file_path)


def test_download_True(get_fake_storage_client, upload_blob):

    [bucket_name, file_path, blob_name, msg] = upload_blob

    downloader = Downloader(get_fake_storage_client)
    file_path = file_path+'_return'
    downloader.download(bucket_name,blob_name,file_path)

    with open(file_path, mode="rb") as f:
        return_msg = f.read()

    assert  return_msg == msg


def test_list_blobs(get_fake_storage_client, upload_blob):

    [bucket_name, _, _, _] = upload_blob

    downloader = Downloader(get_fake_storage_client)
    blobs = downloader.list_blobs(bucket_name)
    assert isinstance(blobs, list) and \
        all(isinstance(blob, Blob) for blob in blobs)

@pytest.mark.parametrize("bucket_name", ["test_unittest", "test-bucket"])
def test_list_blobs_empty(get_fake_storage_client, bucket_name):

    downloader = Downloader(get_fake_storage_client)
    blobs = downloader.list_blobs(bucket_name)
    assert blobs == []


def test_list_blobs_typeerror(get_fake_storage_client):

    with pytest.raises(TypeError):
        downloader = Downloader(get_fake_storage_client)
        downloader.list_blobs(1)


def test_list_buckets(get_fake_storage_client):
    downloader = Downloader(get_fake_storage_client)
    buckets=downloader.list_buckets()
    assert isinstance(buckets, list) and \
        all(type(bucket) is Bucket for bucket in buckets)