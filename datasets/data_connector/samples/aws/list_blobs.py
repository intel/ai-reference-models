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
from data_connector.aws.connector import Connector
from data_connector.aws.downloader import Downloader

# specify a S3 bucket name
bucket_name = 'YOUR_BUCKET_NAME'
# create a connector
connector = Connector()
# connect to aws using default AWS access keys
# connect() method uses the configurations settings for AWS account
conection_object = connector.connect()
# list files from bucket
# create a downloader to list files
downloader = Downloader(conection_object)
# use the list_blobs function
list_blobs = downloader.list_blobs(bucket_name)
# list_blobs function returns all object names in the bucket
print(list_blobs)