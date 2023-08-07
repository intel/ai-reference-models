# Cloud Data Connector

[![Intel: AI](https://img.shields.io/badge/intel-AI-0071C5)](https://www.intel.com/content/www/us/en/developer/topic-technology/artificial-intelligence/overview.html)
[![Python](https://img.shields.io/badge/Python-3.9/3.10-green)](https://www.intel.com/content/www/us/en/developer/topic-technology/artificial-intelligence/overview.html)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![security: SNYK](https://img.shields.io/badge/Security-SNYK-yellow)](https://github.com/PyCQA/bandit)
[![security: BDBA](https://img.shields.io/badge/Security-BDBA-yellow)](https://github.com/PyCQA/bandit)
[![security: Checkmarks](https://img.shields.io/badge/Security-Checkmarks-yellow)](https://github.com/PyCQA/bandit)


_____
## Overview
---
**Cloud data connector** is a tool to connect to AzureML, Azure blob, GCP storage, GCP Big Query  and AWS storage S3. 
The goal is provide all cloud managers in one place and provide documentation for an easy integration.

For more details, visit the [Cloud Data Connector](https://github.com/IntelAI/models/tree/master/datasets/cloud_data_connector) GitHub repository.
<br/><br/>

## Hardware Requirements
---
The hardware should comply with the same requirements that the cloud service.
<br/><br/>

## How it Works
---
The package contains the following modules:

| Package Components   |
| -------------------- |
| cloud_data_connector.aws   |
| cloud_data_connector.gcp   |
| cloud_data_connector.azure |

Each module is capable of connect, download and upload operation to it-s corresponding cloud service.
<br/><br/>

## Getting Started with **Cloud Data Connector**
---

It is strongly recommended to use a virtual environment to ensure proper operation, example:
```bash
conda create -n venv python=3.10 -c conda-forge
conda activate venv
```

You can install the package with:
```bash
python -m pip install cloud-data-connector
```

Please follow module specific documentation for use case, hands-examples.
1. cloud_data_connector/azure/README.md
2. cloud_data_connector/azure/AzureML.md
3. cloud_data_connector/aws/README.md
4. cloud_data_connector/gcp/README.md
<br/><br/>

## Getting Started with **Cloud Data Connector Azure**
---

### Abstract
Intel's [Cloud Data Connector](https://github.com/IntelAI/models/tree/master/datasets/cloud_data_connector/README.md) AzureML tool allows users follow a simple flow to work locally on model training and upload jobs to AzureML configuring a job, workflow, upload models or training scripts.
Data connector for Azure is a tool to connect to Azure Blob and Azure Machine Learning tools.

### Requirements
This tool requires an account on Azure Cloud and an Azure ML worskspace active.
* Azure Account 
* Access to AzureML
* [Create a workspace of Azure ML](https://learn.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources#create-the-workspace)
* [Get configuration file](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-environment#local-and-dsvm-only-create-a-workspace-configuration-file)


### Authentication
Authentication with Azure requires to install [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/authenticate-azure-cli)
```bash
>az login
```
This command load certificates to login using simple authentication method. 


### Storage
All Azure ML Workspaces has an storage blob, with this tool is possible connect to Azure Blob or Azure ML Storage. 

### Azure ML
Azure ML is a service to train models connecting data sources and deploy it in production using Azure infrastructure.

Data connector provides a tool to connect to Azure ML workspaces and upload configurations for this proposes.

### Sample
**Blob Connector**
```python
    from cloud_data_connector.azure import connect
    connection_string_sample = """
        DefaultEndpointsProtocol=http;
        AccountName=<YOUR_ACCOUNT_NAME>;
        AccountKey=<YOUR_ACCOUNT_KEY>;
        BlobEndpoint=http://127.0.0.1:10000/
        devstoreaccount1;
        QueueEndpoint=http://127.0.0.1:10001/
        devstoreaccount1;
        TableEndpoint=http://127.0.0.1:10002/
        devstoreaccount1;
       """
    connector = connect(connection_string=connection_string_sample)

```
How to get a [Connection String](https://learn.microsoft.com/en-us/answers/questions/1071173/where-can-i-find-storage-account-connection-string)? 

![Azure Connection String Sample](https://github.com/IntelAI/models/blob/master/datasets/data_connector/docs/img/connection_string.png)

Also you can get connection strings using Azure CLI
```bash
>az storage account show-connection-string --name <storageAccount> --resource-group <resourceGroup> --subscription <subscription>
```
Or just 
```bash
>az storage account show-connection-string --name <storageAccount> 
```
* This process not works for WSL 


**Blob upload**
```python
    from cloud_data_connector.azure import Uploader

    uploader = Uploader(connector= connector)
    uploader.upload(
        'sample.txt',
        blob_container_name='sample_container'
    )
```

**Blob downloader**
```python
    from cloud_data_connector.azure import Downloader

    downloader = Downloader(connector=connector)
    downloader.download()
```
[See sample Here](https://github.com/IntelAI/models/blob/master/datasets/cloud_data_connector/samples/azure/blob_sample.py)
<br/><br/>


## Getting Started with **Cloud Data Connector AWS**
---
### Cloud Data Connector AWS S3 

Cloud Data Connector for AWS S3 allows you to connect to S3 buckets and list contents, download and upload files.

### Access S3 buckets

To access S3 buckets, you will need to sign up for an AWS account and create access keys. 

Access keys consist of an access key ID and secret access key, which are used to sign programmatic requests that you make to AWS.

### Hot to get your access key ID and secret access key

1. Open the IAM console at https://console.aws.amazon.com/iam/.
2. On the navigation menu, choose Users.
3. Choose your IAM username.
4. Open the Security credentials tab, and then choose Create access key.
5. To see the new access key, choose Show. Your credentials look like the following:
    - Access key ID: my_access_key
    - Secret access key: my_secret_key

### Configuration settings using environment variables for AWS account

You must configure your AWS credentials using environment variables.

By default, you need the next environment variables listed below.

- AWS_ACCESS_KEY_ID: The access key for your AWS account.
- AWS_SECRET_ACCESS_KEY: The secret key for your AWS account.

You can add more configuration settings listed [here](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html#using-environment-variables). For example, you can set the `AWS_SESSION_TOKEN`, it is only needed when you are using temporary credentials.

### Usage

You need to import the DataConnector class.

```python
from cloud_data_connector.aws.connector import Connector
```

Connector class has the method connect(), it creates an AWS S3 object, by default the function will create a S3 connector using the credentials saved in your environment variables.

```python
connector = Connector()
```

Call the connect() method, this will return a connection object for S3. 

```python
conection_object = connector.connect()
```

Import the Downloader class and use the connection object returned by connect() function.

```python
from cloud_data_connector.aws.downloader import Downloader

downloader = Downloader(conection_object)
```

The Downloader class has two methods:

- list_blobs(container_obj): The function to get a list of the objects in a bucket.
- download(container_obj, data_file, destiny): The function to download a file from a S3 bucket.

A first step with buckets is to list their content using the `list_blobs(container_obj)` method. Specify the next parameter.

- container_obj: The bucket name to list.

```python
from cloud_data_connector.aws.downloader import Downloader

downloader = Downloader(conection_object)

list_blobs = downloader.list_blobs('MY_BUCKET_NAME')
print(list_blobs)
```

To download a file use the `download(container_obj, data_file, destiny)` method and specify the next parameters.

- container_obj: The name of the bucket to download from.
- data_file: The name of the file to download from.
- destiny: The path to the file to download to.

```python
from cloud_data_connector.aws.downloader import Downloader

downloader = Downloader(conection_object)
file_name = "path/to_file.csv"
downloader.download(bucket_name, file_name, 'path/to_destiny.csv')
```

You can import an Uploader class and use the upload method to send a file from you local machine to a bucket. You need to add the connector object to Uploader constructor.

```python
from cloud_data_connector.aws.uploader import Uploader
from cloud_data_connector.aws.connector import Connector

connector = Connector()
conection_object = connector.connect()
uploader = Uploader(conection_object)

```
Specify the next parameters in upload function.

- container_obj: The name of the bucket to upload to.
- data_file: The path to the file to upload.
- object_name: The name of the file to upload to.

```python
from cloud_data_connector.aws.uploader import Uploader

uploader = Uploader(conection_object)
uploader.upload(bucket_name, 'path/to_local_file.csv', 'path/to_object_name.csv')
```

### List objects in a bucket

```python
# import the dataconnector package
from cloud_data_connector.aws.connector import Connector
from cloud_data_connector.aws.downloader import Downloader

# specify a S3 bucket name
bucket_name = 'MY_BUCKET_NAME'
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

```

### Download a file

```python
# import the dataconnector package
from cloud_data_connector.aws.connector import Connector
from cloud_data_connector.aws.downloader import Downloader

# specify a S3 bucket name
bucket_name = 'MY_BUCKET_NAME'
# create a connector
connector = Connector()
# connect to aws using default aws access keys
conection_object = connector.connect()
# download a file from bucket
# create a Downloader object using a connector object
downloader = Downloader(conection_object)
# specify the object name to download
file_name = "path/to_file.csv"
# download the object
downloader.download(bucket_name, file_name, 'path/to_destiny.csv')
```

### Upload a file

```python
# import dataconnector package
from cloud_data_connector.aws.connector import Connector
from cloud_data_connector.aws.uploader import Uploader

# specify a S3 bucket name
bucket_name = 'MY_BUCKET_NAME'
# create a connector
connector = Connector()
# connect to aws using default aws access keys
conection_object = connector.connect()
# Upload a file
# create a uploader object using a connection object
uploader = Uploader(conection_object)
# upload a file
uploader.upload(bucket_name, 'path/to_local_file.csv', 'path/to_object_name.csv')
```
<br/><br/>

## Getting Started with **Cloud Data Connector GCP**
---
### GCP Permissions
To enable permissions in GCP to use Storage and BigQuery, from the the left side navigation menu, inside Google Cloud, go to "APIs & Services > Library" and search and enable:
- Cloud Storage
- Google Cloud Storage JSON API
- BigQuery API

You need one of two authentication tools: Oauth or Service Account
### Oauth
To enable oauth2, from the the left side navigation menu go to "APIs & Services > Credentials" and select "+ CREATE CREDENTIALS". From the three available options select "OAuth client ID". When creating an ID it is necessary to provide an "Application type" and a Name, which for this case select "Desktop app" for  "Application type" and type a name of your preference. Once the ID has been created a window will pop with the Client ID and Secret; select "DOWNLOAD JSON" and store the JSON file in a secure place.
### Service Account
To enable a service account, from the left side navigation menu go to "APIs & Services > Credentials" and select "+ CREATE CREDENTIALS". From the three available options select "Service account". In "Service account details" provide a service account name and ID and select "CREATE AND CONTINUE". In "Grant this service account access to project" select roles: "Cloud Storage > Storage Admin" and "BigQuery > BigQuery Admin". After defining roles press "Done". Go to the created service account by selecting the service e-mail in "Service Accounts" and go to "KEYS", from there you can create a new key for the account(it will provide a JSON file).

### Installing Into Docker Container
To be able to run data connector inside a Docker container execute the following command changing options as needed:
```bash
docker run  -it  --name <container_name> --net=host -v <path to frameworks.ai.models.intel-mode.data-connector>:/workspace/model-zoo --env HTTPS_PROXY=$HTTPS_PROXY --env no_proxy=$no_proxy --env HTTP_PROXY=$HTTP_PROXY --env http_proxy=$http_proxy  --entrypoint bash conda/miniconda3:latest
```

Inside the docker container set source to activate conda:
```bash
source /usr/local/bin/activate
```
To install requirements.txt file, first a conda environment must be created:
```bash
conda create -n cloud_data_connector python=3.9 -y && \
conda activate cloud_data_connector
```
To be able to run the GCP part of the connector, GCP CLI must be [installed](https://cloud.google.com/sdk/docs/install). The easy way to install the CLI repository is by running the following commands inside the container:
```bash
apt-get update && \
apt-get install apt-transport-https ca-certificates gnupg curl gpg -y && \
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && apt-get update -y && apt-get install google-cloud-cli -y
```
If apt-key command is not supported use the following command instead:
```bash
apt-get update && \
apt-get install apt-transport-https ca-certificates gnupg curl gpg -y && \
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | tee /usr/share/keyrings/cloud.google.gpg && apt-get update -y && apt-get install google-cloud-sdk -y
```
To initialize the CLI you must have a GCP account. Run the following command, accept the login request, open the provided link in a web browser:
```bash
gcloud init
```
Once in the browser GCP will ask for permissions to access the account; allow the permissions and copy the provided verification code in gcloud CLI. When providing the verification code select the desired cloud project. 

To provide access for the libraries to oauth run:
```bash
gcloud auth application-default login
```
Open the provided link in a web browser. GCP will ask for permissions to access the account; allow the permissions and copy the provided verification code in gcloud CLI.

Go to cloud_data_connector directory, upgrade pip packages and install requirements.txt:
```bash
cd /workspace/model-zoo/cloud_data_connector
pip install --upgrade pip wheel setuptools
pip install -r cloud_data_connector/gcp/requirements.txt 
```

### Running Code Samples
To run code portions inside a container and VS code the port to be used by OAuth must be added to the list of forwarded ports. Open OUTPUT terminal (Ctrl + Shift + U), go to PORTS and select "Add Port".

To provide the JSON key obtained for OAuth in Google Cloud, the file content must be assigned to the 'CLIENT_SECRETS' environment variable. Also it can be obtained from a ".env" file that contains the variable. To use it from a ".env" file "python-dotenv" must be installed and the file must be in a location were "load_dotenv()" can find it. dotenv can find ".env" file inside the directory where a script is being executed, e.g. for the samples provided below you can add ".env" file into ```<base_path_to_model_zoo>/cloud_data_connector/cloud_data_connector/samples/gcp/``` directory.

```bash
pip install python-dotenv
```
To provide the JSON key obtained for the service account, the file path must be provided to the sample scripts if access to Storage or BigCloud is going to be made through service account.

To test GCP Storage a bucket must be created. Go to "Cloud Storage > Buckets" and select "Create". There you need to provide several  options to create a bucket (for the example the name of the bucket will be "dataconnector_data_bucket"), for the following example the default values can be used.

To test GCP Storage there is a script located in ```<base_path_to_model_zoo>/cloud_data_connector/cloud_data_connector/samples/gcp/storage.py```. To run it execute from the base path of data connector for OAuth:
```bash
python -m samples.gcp.storage -o
```
For service account:
```bash
python -m samples.gcp.storage -p <project_name> -c <credentials_path>
```
User must provide the project name using flag (-p) and the local path to the JSON file with the credentials (-c).

To test GCP BigQuery there is a script located in ```<base_path_to_model_zoo>/cloud_data_connector/cloud_data_connector/samples/gcp/bigquery.py```. To run it execute from the base path of data connector for OAuth:
```bash
python -m samples.gcp.bigquery -o
```
For service account:
```bash
python -m samples.gcp.bigquery -p <project_name> -c <credentials_path>
```
User must provide the project name using flag (-p) and the local path to the JSON file with the credentials (-c).
<br/><br/>


## Support
---
If you have questions or issues about this package, contact the [Support Team](mailto:IntelAI@intel.com).
Cloud Data Connector has an Apache license, as found in the [LICENSE](https://www.apache.org/licenses/LICENSE-2.0) file.
Dependencies versions higher than currently implemented ones are in beta and should be used with caution.