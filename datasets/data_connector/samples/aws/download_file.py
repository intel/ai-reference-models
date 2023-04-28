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
# connect to aws using default aws access keys
conection_object = connector.connect()
# download a file from bucket
# create a Downloader object using a connector object
downloader = Downloader(conection_object)
# specify the object name to download
file_name = "1937/35633099999_sample.csv"
# download the object
downloader.download(bucket_name, file_name, './new_file.csv')