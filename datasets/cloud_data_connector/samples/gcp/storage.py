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
import tempfile
import argparse
from dotenv import load_dotenv
from cloud_data_connector.gcp.connector import Connector
from cloud_data_connector.gcp.downloader import Downloader
from cloud_data_connector.gcp.uploader import Uploader

def demo_storage_srvac(project_name, credentials_path, bucket_name = "dataconnector_data_bucket"):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
    connector = Connector("storage")
    storage_client = connector.connect(connection_string=project_name)

    msg = b'Goodbye'

    f = tempfile.NamedTemporaryFile(delete=False)
    f.write(msg)
    f.close()

    file_path = f.name
    blob_name = "Test2"

    print("Uploading to storage")
    uploader = Uploader(storage_client)
    uploader.upload_to_bucket(bucket_name, file_path, blob_name)

    file_path = "test2.txt"
    print("Downloading from storage")
    downloader = Downloader(storage_client)
    downloader.download(bucket_name,blob_name,file_path)


def demo_storage_oauth(port, bucket_name = "dataconnector_data_bucket"):
    load_dotenv()

    CLIENT_SECRETS_STR = os.getenv('CLIENT_SECRETS')
    connector = Connector("storage")
    credentials = connector.get_credentials_from_config(CLIENT_SECRETS_STR,port=port)
    storage_client = connector.connect(connector_object=credentials)

    msg = b'Hello'

    f = tempfile.NamedTemporaryFile(delete=False)
    f.write(msg)
    f.close()

    file_path = f.name
    blob_name = "Test1"

    print("Uploading to storage")
    uploader = Uploader(storage_client)
    uploader.upload_to_bucket(bucket_name, file_path, blob_name)

    file_path = "test1.txt"
    print("Downloading from storage")
    downloader = Downloader(storage_client)
    downloader.download(bucket_name,blob_name,file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--oauth', action='store_true')
    parser.add_argument('-p', '--project', type=str)
    parser.add_argument('-c', '--credentials', type=str)
    args = parser.parse_args()

    if args.oauth:
        demo_storage_oauth(port=9023)
    else:
        demo_storage_srvac(project_name=args.project, credentials_path=args.credentials)
