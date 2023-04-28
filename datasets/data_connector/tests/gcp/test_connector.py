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
from google.cloud.client import ClientWithProject
from google.cloud.storage.client import Client as Client_st
from google.cloud.bigquery.client import Client as Client_bq
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from data_connector.gcp.connector import Connector
from json.decoder import JSONDecodeError


def test_init_valueerror():
    with pytest.raises(ValueError):
        Connector("")


def test_connect_no_credentials():
    with pytest.raises(TypeError):
        con = Connector()
        cred = Credentials(toke="")
        con.connect(connector_object=cred)


def test_connect_none():
    with pytest.raises(NotImplementedError):
        print("Test Connect with None")
        con = Connector()
        con.connect()


@pytest.mark.parametrize("option, client_type", \
                        [("bigquery", Client_bq), ("storage", Client_st)])
def test_connect_type_oath(option, client_type, mocker):
    #Patches oauth
    mocker.patch.object(ClientWithProject, "_determine_default", return_value="test_unittest")

    con = Connector(option)
    cred = Credentials(token="test_unittest")
    client = con.connect(connector_object=cred)
    assert isinstance(client, client_type)


@pytest.mark.parametrize("option, client_type", \
                        [("bigquery", Client_bq), ("storage", Client_st)])
@pytest.mark.skip
# Skipped because it needs browser login, and return the generated key, which would take time to automatize.
def test_connect_str(option, client_type):
    con = Connector(option)
    client = con.connect(connection_string='test')
    assert isinstance(client, client_type)


@pytest.mark.parametrize("option, client_type", \
                        [("bigquery", Client_bq), ("storage", Client_st)])
def test_connect_str_error(option, client_type):
    with pytest.raises(NotImplementedError):
        con = Connector(option)
        client = con.connect(connection_string="")
        assert isinstance(client, client_type)


@pytest.mark.parametrize("key_file_path, port, timeout_seconds", \
                        [(1,1,1), ("test_unittest", 1, "test_unittest"), ("test_unittest", "test_unittest", 1)])
def test_credentials_file_exceptions_1(key_file_path, port, timeout_seconds):
    with pytest.raises(TypeError):
        con = Connector()
        con.get_credentials_from_file(key_file_path, port, timeout_seconds)


@pytest.mark.parametrize("key_file_path, port, timeout_seconds", \
                        [("test_unittest", -1, 1), ("test_unittest", 1, -1)])
def test_credentials_file_exceptions_2(key_file_path, port, timeout_seconds):
    with pytest.raises(ValueError):
        con = Connector()
        con.get_credentials_from_file(key_file_path, port, timeout_seconds)


def test_credentials_file_exceptions_3():
    with pytest.raises(FileNotFoundError):
        con = Connector()
        con.get_credentials_from_file("test_unittest", 1, 1)


def test_credentials_file_exceptions_4(mocker):
    with pytest.raises(AttributeError):
        mocker.patch("os.path.isfile", return_value=True)
        mocker.patch("google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file", return_value=InstalledAppFlow)
        mocker.patch.object(InstalledAppFlow, "run_local_server", side_effect=AttributeError)
        con = Connector()
        con.get_credentials_from_file("test_unittest", 1, 1)


@pytest.mark.parametrize("key_str, port, timeout_seconds", \
                        [(1,1,1), ("test_unittest", 1, "test_unittest"), ("test_unittest", "test_unittest", 1)])
def test_credentials_config_exceptions_1(key_str, port, timeout_seconds):
    with pytest.raises(TypeError):
        con = Connector()
        con.get_credentials_from_config(key_str, port, timeout_seconds)


@pytest.mark.parametrize("key_str, port, timeout_seconds", \
                        [("test_unittest", -1, 1), ("test_unittest", 1, -1)])
def test_credentials_config_exceptions_2(key_str, port, timeout_seconds):
    with pytest.raises(ValueError):
        con = Connector()
        con.get_credentials_from_config(key_str, port, timeout_seconds)


def test_credentials_config_exceptions_3():
    with pytest.raises(FileNotFoundError):
        con = Connector()
        con.get_credentials_from_config("test_unittest", 1, 1)


def test_credentials_config_exceptions_4(mocker):
    json_str = '{"test_unittest": ""}'
    with pytest.raises(AttributeError):
        mocker.patch("os.path.isfile", return_value=True)
        mocker.patch("google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file", return_value=InstalledAppFlow)
        mocker.patch.object(InstalledAppFlow, "run_local_server", side_effect=AttributeError)
        con = Connector()
        con.get_credentials_from_file(json_str, 1, 1)
