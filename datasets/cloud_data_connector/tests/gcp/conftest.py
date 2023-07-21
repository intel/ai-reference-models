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
import pytest
import tempfile
from google.cloud import storage
from gcp_storage_emulator.server import create_server
from cloud_data_connector.gcp.uploader import Uploader

plugins = ["pytest-bigquery-mock"]

@pytest.fixture()
def get_fake_storage_client():

    HOST = "localhost"
    PORT = 9023
    BUCKET = "test-bucket"

    # default_bucket parameter creates the bucket automatically
    server = create_server(HOST, PORT, in_memory=True, default_bucket=BUCKET)
    server.start()

    os.environ["STORAGE_EMULATOR_HOST"] = f"http://{HOST}:{PORT}"
    client = storage.Client()
    yield client

    server.stop()


@pytest.fixture()
def upload_blob(get_fake_storage_client):
    msg = b'MtW'

    f = tempfile.NamedTemporaryFile(delete=False)
    f.write(msg)
    f.close()

    bucket_name = "test-bucket"
    file_path = f.name
    blob_name = "MtW Test"

    uploader = Uploader(get_fake_storage_client)
    uploader.upload_to_bucket(bucket_name, file_path, blob_name)

    return [bucket_name, file_path, blob_name, msg]
