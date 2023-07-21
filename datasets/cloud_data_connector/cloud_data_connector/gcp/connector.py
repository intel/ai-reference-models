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
from cloud_data_connector.int.int_connector import IntConnector
from google.oauth2.credentials import Credentials
from google.cloud.client import Client
from google_auth_oauthlib import flow
from google.cloud import bigquery
from google.cloud import storage
from abc import ABCMeta
import json
from json import JSONDecodeError
import os

class Connector(IntConnector, metaclass=ABCMeta):
    # This OAuth 2.0 access scope allows for full read/write access to the
    # authenticated user's account.
    __SCOPES = ['https://www.googleapis.com/auth/bigquery', \
                'https://www.googleapis.com/auth/devstorage.read_write']

    def __init__(self, credential_type:str="bigquery") -> None:
        super().__init__()
        self.scope =[]
        self.__connector = None

        if credential_type=="bigquery" or credential_type=="storage":
            if credential_type =="bigquery":
                self.scope.append(self.__SCOPES[0])
            elif credential_type =="storage":
                self.scope.append(self.__SCOPES[1])
            
            self.credential_type=credential_type
        else:
            raise ValueError("Valid credential_type's are bigquery or storage, "\
                            "received {credential_type}")


    def connect(self, connection_string:str="", connector_object:object=None)-> object:
        if not (isinstance(connector_object, Credentials) or (isinstance(connection_string, str) and connection_string!="")):
            raise NotImplementedError("Connection type not supported")
        else:
            return self.connect_gcp(credentials_object=connector_object, project_name=connection_string)


    def connect_gcp(self, credentials_object:Credentials=None,  project_name:str="") -> Client:
        if credentials_object != None:
            if isinstance(credentials_object,Credentials):
                if self.credential_type=="bigquery":
                    self.__connector = bigquery.Client(credentials=credentials_object)
                elif self.credential_type=="storage":
                    self.__connector = storage.Client(credentials=credentials_object)
                
                return self.__connector
            else:
                raise TypeError("Must provide credentials_object of type Credentials "\
                                "instead of type {}".format(type(credentials_object)))
        elif project_name != "":
            if isinstance(project_name, str):
                if self.credential_type=="bigquery":
                    self.__connector = bigquery.Client(project=project_name)
                elif self.credential_type=="storage":
                    self.__connector = storage.Client(project=project_name)
                
                return self.__connector
            else:
                raise TypeError("Must provide project_name of type str " \
                                "instead of type {}".format(type(credentials_object)))
        else:
            raise ValueError("Must provide credentials_object or project_name")


    def get_connector(self):
        return self.__connector


    def get_credentials_from_file(self, key_file_path:str, port:int=8088, timeout_seconds:int=20)-> Credentials:
        if not isinstance(key_file_path, str):
            raise TypeError("key_file_path must be str.")
        if not isinstance(port, int):
            raise TypeError("port must be int.")
        if not isinstance(timeout_seconds, int):
            raise TypeError("timeout_seconds must be int.")
        
        if port < 0:
            raise ValueError('port should be a positive number.')
        if timeout_seconds < 0:
            raise ValueError('timeout_seconds should be a positive number')
        
        if not os.path.isfile(key_file_path):
            raise FileNotFoundError("{} not found.".format(key_file_path))
        
        credentials = None
        try:

            appflow = flow.InstalledAppFlow.from_client_secrets_file(
                key_file_path,
                scopes=self.scope)

            credentials = appflow.run_local_server(
                host='localhost',
                port=port,
                authorization_prompt_message='Please visit this URL: {url}',
                success_message='The auth flow is complete; you may close this window.',
                open_browser=True,
                timeout_seconds=timeout_seconds)
        
        except AttributeError as e:
            print(e)
            raise AttributeError("Wrong type of attribute, credentials is type {}.\n\
                If type None, this might be caused due to timeout.".format(type(credentials)))
        except OSError as e :
            print(e)
            raise OSError("If flow server didn't stop correctly, wait 60 seconds " \
                        "for it to stop.")

        return credentials
    

    def get_credentials_from_config(self, key_str:str, port:int=8088, timeout_seconds:int=20)-> Credentials:

        if not isinstance(key_str, str):
            raise TypeError("key_file_path must be str.")
        elif not isinstance(port, int):
            raise TypeError("port must be int.")
        elif not isinstance(timeout_seconds, int):
            raise TypeError("timeout_seconds must be int.")
        
        if port < 0:
            raise ValueError('port should be a positive number.')
        elif timeout_seconds < 0:
            raise ValueError('timeout_seconds should be a positive number')

        credentials = None
        
        try:
            try:
                _key_str = json.loads(key_str)
            except JSONDecodeError as jse:
                with open(key_str) as kf:
                    _key_str = json.load(kf)
            key_str = _key_str
        except FileNotFoundError as e:
            print(e)
            raise FileNotFoundError


        try: 
            appflow = flow.InstalledAppFlow.from_client_config(
                key_str,
                scopes=self.scope)

            credentials = appflow.run_local_server(
                host='localhost',
                port=port,
                authorization_prompt_message='Please visit this URL: {url}',
                success_message='The auth flow is complete; you may close this window.',
                open_browser=True,
                timeout_seconds=timeout_seconds)

        except AttributeError as e:
            print(e)
            raise AttributeError("Wrong type of attribute, credentials is type {}.\n" \
                "If type None, this might be caused due to timeout.".format(type(credentials)))
        except OSError as e :
            print(e)
            raise OSError("If flow server didn't stop correctly, wait 60 seconds " \
                        "for it to stop.")

        return credentials
