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
import unittest
from unittest.mock import MagicMock, patch
import pytest
import os
from dotenv import load_dotenv
# Data type
from azure.storage.blob import BlobServiceClient
# Class to test
from cloud_data_connector.azure.connector import (
    Connector,
    CONFIGURATION_FILE_MISSED_MSG,
    CONNECTION_ERROR_MSG,
    )


load_dotenv()

class TestConnector(unittest.TestCase):
    def setUp(self):
        load_dotenv()
        self.az_connectionstring = os.getenv('AZURE_STORAGE_CONNECTION_STRING', "blob.core")
        self.az_blob_container = os.environ.get('AZURE_BLOB_CONTAINER', 'data-connector')
        self.az_blob_account_name = os.environ.get('AZURE_STORAGE_ACCOUNT_NAME', 'dataconnector0653512919')
        self.connector = Connector() 

    def test_connection(self):
        self.assertIsNotNone(self.az_connectionstring)
        self.connector.connect = MagicMock(return_value=BlobServiceClient( self.az_connectionstring))
        connection = self.connector.connect(self.az_connectionstring)
        self.assertIsInstance(connection, BlobServiceClient )
        

    def test_connection_exception(self):
        with self.assertRaises(ConnectionError) as ce:
            self.connector.connect_by_connection_string = MagicMock(return_value=None)
            self.connector.connect( self.az_connectionstring,"x")
            self.assertEqual(CONNECTION_ERROR_MSG, ce.msg)
    
    
    def test_connect_from_config_file(self):
        with self.assertRaises(FileNotFoundError) as fnfe:
            self.connector.connect_from_config_file(conf_file="")
            self.assertEqual(CONFIGURATION_FILE_MISSED_MSG, fnfe.msg)
        
        