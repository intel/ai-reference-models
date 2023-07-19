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
from google.api_core.exceptions import NotFound
from time import sleep
from google.cloud import bigquery
from typing import List, Tuple

class Query():

    def __init__(self, client:bigquery.Client):
        if not isinstance(client, bigquery.Client):
            raise TypeError("client must be bigquery.Client")
        self.client = client


    def dataset_exist(self, dataset_ref:bigquery.DatasetReference) -> Tuple[bool, bigquery.Dataset]:

        if not isinstance(dataset_ref,bigquery.DatasetReference):
            raise TypeError("dataset_ref must be of type bigquery.DatasetReference")

        try:
            dataset = self.client.get_dataset(dataset_ref)
            return (True, dataset)
        except NotFound as e:
            print(e)
            return (False, None)


    def table_exist(self, table_ref:bigquery.TableReference) -> Tuple[bool, bigquery.Table]:

        if not isinstance(table_ref,bigquery.TableReference):
            raise TypeError("table_ref must be of type bigquery.TableReference")
        
        try:
            table = self.client.get_table(table_ref) 
            return (True, table)
        except NotFound:
            return (False, None)


    def get_datasetref(self, dataset_name:str) -> bigquery.DatasetReference:

        if not isinstance(dataset_name, str):
            raise TypeError("dataset_name must be str")

        dataset_ref = self.client.dataset(dataset_name)
        
        return dataset_ref


    def get_tableref(self, dataset_ref:bigquery.DatasetReference, table_name:str) -> bigquery.TableReference:

        if not isinstance(dataset_ref, bigquery.DatasetReference):
            raise TypeError("dataset_ref must be bigquery.DatasetReference")

        if not isinstance(table_name, str):
            raise TypeError("table_name must be str")

        table_ref = dataset_ref.table(table_name)

        return table_ref


    def create_dataset(self, dataset_name:str, location:str='US') -> bigquery.Dataset:

        if not isinstance(dataset_name, str):
            raise TypeError("dataset_name must be str")

        if not isinstance(location, str):
            raise TypeError("location must be str")
        
        dataset_ref = self.get_datasetref(dataset_name)

        exists, dataset = self.dataset_exist(dataset_ref)

        if exists:
            print('Dataset {} already exists.'.format(dataset.dataset_id))
        else:
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = location
            dataset = self.client.create_dataset(dataset)
            print('Dataset {} created.'.format(dataset.dataset_id))
            
        return dataset


    def create_table(self, dataset_name:str, table_name:str, schema:List[bigquery.SchemaField]) -> bigquery.Table:

        if not isinstance(dataset_name, str):
            raise TypeError("dataset_name must be str")

        if not isinstance(table_name, str):
            raise TypeError("table_name must be str")

        if not all(isinstance(elem,bigquery.SchemaField) for elem in schema):
            raise TypeError("schema must be [bigquery.SchemaField]")
        

        dataset_ref = self.get_datasetref(dataset_name)
        table_ref = self.get_tableref(dataset_ref, table_name)

        exists, table = self.table_exist(table_ref)

        if exists:
            print('Table {} already exists.'.format(table.table_id))
        else:
            print(schema)
            table = bigquery.Table(table_ref, schema=schema)
            table = self.client.create_table(table)
            print('Table {} created.'.format(table.table_id))

        return table


    def delete_table(self, dataset_name:str, table_name:str):

        if not isinstance(dataset_name, str):
            raise TypeError("dataset_name must be str")

        if not isinstance(table_name, str):
            raise TypeError("table_name must be str")

        dataset_ref = self.get_datasetref(dataset_name)
        table_ref = self.get_tableref(dataset_ref, table_name)

        exists, table = self.table_exist(table_ref)

        if exists:
            self.client.delete_table(table)
            print('Table {} deleted.'.format(table.table_id))
        else:
            print("Table {} doesn't exist.".format(table.table_id))  


    def delete_dataset(self, dataset_name:str):

        if not isinstance(dataset_name, str):
            raise TypeError("dataset_name must be str")

        dataset_ref = self.get_datasetref(dataset_name)
        exists, dataset = self.dataset_exist(dataset_ref)

        if exists:
            self.client.delete_dataset(dataset)
            print('Dataset {} deleted.'.format(dataset.dataset_id))
        else:
            print("Dataset {} doesn't exist.".format(dataset.dataset_id))


    def export_items_to_bq(self, dataset_name:str, table_name:str, rows_to_insert:List[Tuple], max_tries=15, pause=3):    
        
        if not isinstance(dataset_name, str):
            raise TypeError("dataset_name must be str")

        if not isinstance(table_name, str):
            raise TypeError("table_name must be str")
        
        if not all(isinstance(elem,Tuple) for elem in rows_to_insert):
            raise TypeError("rows_to_insert must be [Tuple]")
        
        dataset_ref = self.get_datasetref(dataset_name)
        table_ref = self.get_tableref(dataset_ref, table_name)
        exists, table = self.table_exist(table_ref)

        if exists:
            try:
                dataset_ref = self.client.dataset(dataset_name)                                                                                                                            
                table_ref = dataset_ref.table(table_name)   
                table = self.client.get_table(table_ref)

                tries = 0
                while True: #Prevents insert_rows from failing when table was recently created
                    try:
                        errors = self.client.insert_rows(table, rows_to_insert)
                    except NotFound as e:
                        errors = e
                        if tries < max_tries:
                            tries += 1
                            print("Insert rows attempt #{}".format(tries))
                            sleep(pause)
                            continue
                    break

                if errors == []:
                    print("Rows inserted correctly to table {}.".format(table_ref.table_id))
                else:
                    print(errors)

            except ValueError as e:
                print(e)
        else:
            print('Table {} reference not found.'.format(table_ref.table_id))


    def sql_query(self, sql_query:str) -> bigquery.QueryJob:
        
        query_job = self.client.query(sql_query)
        
        return query_job