# Cloud Data Connector: AWS S3 

Cloud Data Connector for AWS S3 allows you to connect to S3 buckets and list contents, download and upload files.

## Access S3 buckets

To access S3 buckets, you will need to sign up for an AWS account and create access keys. 

Access keys consist of an access key ID and secret access key, which are used to sign programmatic requests that you make to AWS.

How to get your access key ID and secret access key
---

1. Open the IAM console at https://console.aws.amazon.com/iam/.
2. On the navigation menu, choose Users.
3. Choose your IAM username.
4. Open the Security credentials tab, and then choose Create access key.
5. To see the new access key, choose Show. Your credentials look like the following:
    - Access key ID: my_access_key
    - Secret access key: my_secret_key

## Configuration settings using environment variables for AWS account

You must configure your AWS credentials using environment variables.

By default, you need the next environment variables listed below.

- AWS_ACCESS_KEY_ID: The access key for your AWS account.
- AWS_SECRET_ACCESS_KEY: The secret key for your AWS account.

You can add more configuration settings listed [here](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html#using-environment-variables). For example, you can set the `AWS_SESSION_TOKEN`, it is only needed when you are using temporary credentials.

## Usage

You need to import the DataConnector class.

```python
from cloud_data_connector.aws import Connector as aws_connector
```

Connector class has the method connect(), it creates an AWS S3 object, by default the function will create a S3 connector using the credentials saved in your environment variables.

```python
aws_bucket_connector = aws_connector()
```

Call the connect() method, this will return a connection object for S3. 

```python
aws_conection_object = aws_bucket_connector.connect()
```

Downloader
---
Import the Downloader class and use the connection object returned by connect() function.


```python
from cloud_data_connector.aws import Downloader as aws_downloader

aws_file_downloader = aws_downloader(aws_conection_object)
```



To download a file use the `download(container_obj, data_file, destiny)` method and specify the next parameters.

- container_obj: The name of the bucket to download from.
- data_file: The name of the file to download from.
- destiny: The path to the file to download to.

```python
from cloud_data_connector.aws import Downloader as aws_downloader

downloader = aws_downloader(aws_conection_object)
file_name = "path/to_file.csv"
downloader.download(bucket_name, file_name, 'path/to_destiny.csv')
```
List Blobs
---
[Downloader](#downloader) class has two methods:
- list_blobs(container_obj): The function to get a list of the objects in a bucket.
- download(container_obj, data_file, destiny): The function to download a file from a S3 bucket.

A first step with buckets is to list their content using the `list_blobs(container_obj)` method. Specify the next parameter.

- container_obj: The bucket name to list.

```python
from cloud_data_connector.aws import Downloader as aws_downloader

aws_bucket_downloader = aws_downloader(aws_conection_object)

list_blobs = aws_bucket_downloader.list_blob(
    'MY_BUCKET_NAME'
)
print(list_blobs)
```

Upload
---

You can import an Uploader class and use the upload method to send a file from you local machine to a bucket. You need to add the connector object to Uploader constructor.

```python
from cloud_data_connector.aws import Uploader as aws_uploader
from cloud_data_connector.aws  import Connector as aws_connector

aws_bucket_connector = aws_connector()
aws_conection_object = aws_bucket_connector.connect()
aws_bucker_uploader = aws_uploader(aws_conection_object)

```
Specify the next parameters in upload function.

- container_obj: The name of the bucket to upload to.
- data_file: The path to the file to upload.
- object_name: The name of file in cloud bucket

```python
aws_bucker_uploader.upload('<bucket_name>', '<path/to_local_file>', '<path/to_object_name>')
```


Samples
---
### List objects in a bucket

```python
# import the dataconnector package
from cloud_data_connector.aws import Connector as aws_connector
from cloud_data_connector.aws import Downloader as aws_downloader

# specify a S3 bucket name
bucket_name = 'MY_BUCKET_NAME'
# create a connector
aws_bucket_connector = aws_connector()
# connect to aws using default AWS access keys
# connect() method uses the configurations settings for AWS account
aws_conection_object = aws_bucket_connector.connect()
# list files from bucket
# create a downloader to list files
aws_downloader = aws_downloader(aws_conection_object)
# use the list_blobs function
list_blobs = aws_downloader.list_blobs(bucket_name)
# list_blobs functions returns all objects in bucket
print(list_blobs)

```

### Download a file

```python
# import the dataconnector package
from cloud_data_connector.aws import Connector as aws_connector
from cloud_data_connector.aws import Downloader as aws_connector

# specify a S3 bucket name
bucket_name = 'MY_BUCKET_NAME'
# create a connector
aws_bucket_connector = aws_connector()
# connect to aws using default aws access keys
aws_conection_object = aws_bucket_connector.connect()
# download a file from bucket
# create a Downloader object using a connector object
aws_downloader = aws_connector(aws_conection_object)
# specify the object name to download
file_name = "path/to_file.csv"
# download the object
aws_downloader.download(bucket_name, file_name, 'path/to_destiny.csv')
```

### Upload a file

```python
# import dataconnector package
from cloud_data_connector.aws import Connector as aws_connector
from cloud_data_connector.aws import Uploader as aws_uploader

# specify a S3 bucket name
bucket_name = 'MY_BUCKET_NAME'
# create a connector
aws_bucket_connector = aws_connector()
# connect to aws using default aws access keys
aws_conection_object = aws_bucket_connector.connect()
# Upload a file
# create a uploader object using a connection object
aws_bucket_uploader = aws_uploader(aws_conection_object)
# upload a file
aws_bucket_uploader.upload(bucket_name, 'path/to_local_file.csv', 'path/to_object_name.csv')
```