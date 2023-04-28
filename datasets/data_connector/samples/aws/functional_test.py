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

#

# import the dataconnector package
import sys
from data_connector.aws.connector import Connector
from data_connector.aws.downloader import Downloader
from data_connector.aws.uploader import Uploader

# specify a S3 bucket name
bucket_name = sys.argv[1]

# specify file to manipulate
file_name = sys.argv[2]

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
# list_blobs functions returns all objects in bucket
print(list_blobs)

# Upload a file
# create a uploader object using a connection object
uploader = Uploader(conection_object)
# upload a file
uploader.upload(bucket_name, file_name, file_name)

# download the object
downloader.download(bucket_name, file_name, 'file_test_2.txt')
list_blobs = downloader.list_blobs(bucket_name)
print(list_blobs)

# delete an object using the s3 client returned by connector.connect() function
# and use the delete_object() function provided by AWS S3 client conection_object.delete_object()
conection_object.delete_object(Bucket=bucket_name, Key=file_name)
