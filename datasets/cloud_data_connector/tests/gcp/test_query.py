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
from google.cloud.bigquery import Client, DatasetReference, TableReference, SchemaField
from google.api_core.exceptions import NotFound
from cloud_data_connector.gcp.query import Query
from google.cloud.bigquery.dataset import Dataset, Table


def test_query_init():
    with pytest.raises(TypeError):
        Query(client="")


def test_sql_no_dataset(mocker,bq_client_mock):
    bq_client_mock.__class__ = Client #Uses client mock as Client class
    mocker.patch.object(bq_client_mock, "get_dataset", side_effect=NotFound(""))

    query = Query(bq_client_mock)
    assert (False, None) == query.dataset_exist(DatasetReference("test_unittest","test_unittest"))


def test_sql_check_dataset(mocker, bq_client_mock):
    bq_client_mock.__class__ = Client 
    mocker.patch.object(bq_client_mock, "get_dataset", side_effect=Dataset)

    query = Query(bq_client_mock)
    result = query.dataset_exist(DatasetReference("test_unittest","test_unittest"))
    assert result[0] == True and isinstance(result[1], Dataset)


def test_sql_no_table(mocker, bq_client_mock):
    bq_client_mock.__class__ = Client 
    mocker.patch.object(bq_client_mock, "get_table", side_effect=NotFound(""))

    query = Query(bq_client_mock)
    assert (False, None) == query.table_exist(TableReference(DatasetReference("test_unittest","test_unittest"),"test_unittest"))


def test_sql_check_table(mocker, bq_client_mock):
    bq_client_mock.__class__ = Client
    mocker.patch.object(bq_client_mock, "get_table", side_effect=Table)

    query = Query(bq_client_mock)
    result = query.table_exist(TableReference(DatasetReference("test_unittest","test_unittest"),"test_unittest"))
    assert result[0] == True and isinstance(result[1], Table)


def test_get_datasetref(bq_client_mock):
    with pytest.raises(TypeError):
        bq_client_mock.__class__ = Client
        query = Query(bq_client_mock)
        query.get_datasetref(None)


@pytest.mark.parametrize("dataset_ref, table_name", \
                        [(DatasetReference("test_unittest", "test_unittest"),0), \
                        (None,"test_unittest")])
def test_get_tableref(dataset_ref, table_name, bq_client_mock):
    with pytest.raises(TypeError):
        bq_client_mock.__class__ = Client
        query = Query(bq_client_mock)
        query.get_tableref(dataset_ref, table_name)


@pytest.mark.parametrize("dataset_name, location", \
                        [("test_unittest", 10), \
                        (10, "test_unittest")])
def test_create_dataset(dataset_name, location, bq_client_mock):
    with pytest.raises(TypeError):
        bq_client_mock.__class__ = Client
        query = Query(bq_client_mock)
        query.create_dataset(dataset_name, location)


@pytest.mark.parametrize("dataset_name, table_name, schema", \
                        [("test_unittest", 10, [SchemaField('Name', 'STRING', mode='REQUIRED')]), \
                        (10, "test_unittest", [SchemaField('Name', 'STRING', mode='REQUIRED')]), \
                        ("test_unittest", "test_unittest", [None])])
def test_create_table(dataset_name, table_name, schema, bq_client_mock):
    with pytest.raises(TypeError):
        bq_client_mock.__class__ = Client
        query = Query(bq_client_mock)
        query.create_table(dataset_name, table_name, schema)


@pytest.mark.parametrize("dataset_name, table_name", \
                        [(0, "test_unittest"), \
                        ("test_unittest", 0)])
def test_delete_table(dataset_name, table_name, bq_client_mock):
    with pytest.raises(TypeError):
        bq_client_mock.__class__ = Client
        query = Query(bq_client_mock)
        query.delete_table(dataset_name, table_name)


def test_delete_dataset(bq_client_mock):
    with pytest.raises(TypeError):
        bq_client_mock.__class__ = Client
        query = Query(bq_client_mock)
        query.delete_dataset(0)


@pytest.mark.parametrize("dataset_name, table_name, rows_to_insert", \
                        [("test_unittest", 10, [('Test', 'Test', 'Test')]), \
                        (10, "test_unittest", [('Test', 'Test', 'Test')]), \
                        ("test_unittest", "test_unittest", [None])])
def test_export_items_to_bq(dataset_name, table_name, rows_to_insert, bq_client_mock):
    with pytest.raises(TypeError):
        bq_client_mock.__class__ = Client
        query = Query(bq_client_mock)
        query.export_items_to_bq(dataset_name, table_name, rows_to_insert)


@pytest.mark.bq_query_return_data(
    [
        {
            "query": "SELECT * FROM table",
            "table": {
                "columns": [
                    "id_row",
                    "name",
                ],
                "rows": [
                    [1, "Alice"],
                    [2, "Pete"],
                    [3, "Steven"],
                ],
            },
        },
    ]
)
def test_sql_query(bq_client_mock):
    
    bq_client_mock.__class__ = Client # Change class for mock
    query = Query(bq_client_mock)

    expected_row_dicts = [
        {"id_row": 1, "name": "Alice"},
        {"id_row": 2, "name": "Pete"},
        {"id_row": 3, "name": "Steven"},
    ]

    rows = query.sql_query("SELECT * FROM table").result()

    for row, expected_row in zip(rows, expected_row_dicts):
        assert dict(row) == expected_row