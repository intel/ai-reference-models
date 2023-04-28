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
import argparse
from time import sleep
from data_connector.gcp.connector import Connector
from data_connector.gcp.query import Query
from dotenv import load_dotenv
from google.cloud import bigquery

load_dotenv()

dataset_name = "demo_dataset_dataconnector"
table_name = "demo_table_dataconnector"
project_name = "intel-optimized-tensorflow"

def dummy_callback(future):
    global jobs_done
    jobs_done[future.job_id] = True


def make_query(bq:Query):
    global jobs_done

    sql_query = """SELECT
                    Name, S_No, Age_in_cm, Weight_in_Kg
                    FROM `{0}.{1}.{2}`
                    LIMIT 1000;""".format(project_name, dataset_name, table_name)
    query_job1 = bq.sql_query(sql_query)

    sql_query2 = """INSERT INTO 
                    `demo_dataset_dataconnector.demo_table_dataconnector` 
                    VALUES (111, 222, 333, 'Test');"""
    
    query_job2 = bq.sql_query(sql_query2)

    jobs = [query_job1, query_job2]
    jobs_done = {job.job_id: False for job in jobs}

    [job.add_done_callback(dummy_callback) for job in jobs]

    try:
        for row in query_job1.result():
            print(row)
    except BadRequest as e:
        for e in query_job1.errors:
            print('ERROR: {}'.format(e['message']))


def insert_rows(bq):
    rows_to_insert = [
        (1, 32, 32, "Hector"),
        (2, 64, 29, "Joe"),
        (3, 108, 108, "Sandy")
    ]

    bq.export_items_to_bq(dataset_name, table_name, rows_to_insert)


def create_table(bq):

    bq.create_dataset(dataset_name)
    schema = [
        bigquery.SchemaField('S_No', 'INTEGER', mode='REQUIRED'),
        bigquery.SchemaField('Age_in_cm', 'INTEGER', mode='REQUIRED'),
        bigquery.SchemaField('Weight_in_Kg', 'INTEGER', mode='REQUIRED'),
        bigquery.SchemaField('Name', 'STRING', mode='REQUIRED'),
    ]
    bq.create_table(dataset_name, table_name, schema)


def check_datasets(datasets, bigquery_client):
    project = bigquery_client.project
    if datasets:
        print("Datasets in project {}:".format(project))
        for dataset in datasets:
            print("\t{}".format(dataset.dataset_id))
    else:
        print("{} project does not contain any datasets.".format(project))


def demo_bigquery_oauth(port):
    CLIENT_SECRETS_STR = os.getenv('CLIENT_SECRETS')
    connector = Connector("bigquery")
    credentials = connector.get_credentials_from_config(CLIENT_SECRETS_STR,port=port)
    bigquery_client = connector.connect(connector_object=credentials)

    datasets = list(bigquery_client.list_datasets())
    check_datasets(datasets, bigquery_client)

    bq = Query(bigquery_client)

    create_table(bq)
    sleep(5)
    insert_rows(bq)
    sleep(5)
    make_query(bq)


def demo_bigquery_srvac(project_name, credentials_path):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
    connector = Connector("bigquery")
    bigquery_client = connector.connect(connection_string=project_name)

    datasets = list(bigquery_client.list_datasets())
    check_datasets(datasets, bigquery_client)

    bq = Query(bigquery_client)

    create_table(bq)
    insert_rows(bq)
    make_query(bq)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--oauth', action='store_true')
    parser.add_argument('-c', '--credentials', type=str)
    parser.add_argument('-p', '--project', type=str)
    args = parser.parse_args()

    if args.oauth:
        demo_bigquery_oauth(port=9023)
    else:
        demo_bigquery_srvac(project_name=args.project, credentials_path=args.credentials)

