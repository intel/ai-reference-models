# GCP Permissions
To enable permissions in GCP to use Storage and BigQuery, from the the left side navigation menu, inside Google Cloud, go to "APIs & Services > Library" and search and enable:
- Cloud Storage
- Google Cloud Storage JSON API
- BigQuery API

You need one of two authentication tools: Oauth or Service Account
### Oauth
To enable oauth2, from the the left side navigation menu go to "APIs & Services > Credentials" and select "+ CREATE CREDENTIALS". From the three available options select "OAuth client ID". When creating an ID it is necessary to provide an "Application type" and a Name, which for this case select "Desktop app" for  "Application type" and type a name of your preference. Once the ID has been created a window will pop with the Client ID and Secret; select "DOWNLOAD JSON" and store the JSON file in a secure place.
### Service Account
To enable a service account, from the left side navigation menu go to "APIs & Services > Credentials" and select "+ CREATE CREDENTIALS". From the three available options select "Service account". In "Service account details" provide a service account name and ID and select "CREATE AND CONTINUE". In "Grant this service account access to project" select roles: "Cloud Storage -> Storage Admin" and "BigQuery -> BigQuery Admin". After defining roles press "Done". Go to the created service account by selecting the service e-mail in "Service Accounts" and go to "KEYS", from there you can create a new key for the account(it will provide a JSON file).

# Installing Into Docker Container
To be able to run data connector inside a Docker container execute the following command changing options as needed:
```bash
docker run  -it  --name <container_name> --net=host -v <path to IntelAI/models>:/workspace/model-zoo --entrypoint bash conda/miniconda3:latest
```

Inside the docker container set source to activate conda:
```bash
source /usr/local/bin/activate
```
To install requirements.txt file, first a conda environment must be created:
```bash
conda create -n data_connector python=3.9 -y && \
conda activate data_connector
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

Go to data_connector directory, upgrade pip packages and install requirements.txt:
```bash
cd /workspace/model-zoo/data_connector
pip install --upgrade pip wheel setuptools
pip install -r data_connector/gcp/requirements.txt 
```

# Running Code Samples
To run code portions inside a container and VS code the port to be used by OAuth must be added to the list of forwarded ports. Open OUTPUT terminal (Ctrl + Shift + U), go to PORTS and select "Add Port".

To provide the JSON key obtained for OAuth in Google Cloud, the file content must be assigned to the 'CLIENT_SECRETS' environment variable. Also it can be obtained from a ".env" file that contains the variable. To use it from a ".env" file "python-dotenv" must be installed and the file must be in a location were "load_dotenv()" can find it. dotenv can find ".env" file inside the directory where a script is being executed, e.g. for the samples provided below you can add ".env" file into <base_path_to_model_zoo>/data_connector/data_connector/samples/gcp/ directory.

```bash
pip install python-dotenv
```
To provide the JSON key obtained for the service account, the file path must be provided to the sample scripts if access to Storage or BigCloud is going to be made through service account.

To test GCP Storage a bucket must be created. Go to "Cloud Storage > Buckets" and select "Create". There you need to provide several  options to create a bucket (for the example the name of the bucket will be "dataconnector_data_bucket"), for the following example the default values can be used.

To test GCP Storage there is a script located in <base_path_to_model_zoo>/data_connector/data_connector/samples/gcp/storage.py. To run it execute from the base path of data connector for OAuth:
```bash
python -m samples.gcp.storage -o
```
For service account:
```bash
python -m samples.gcp.storage -p <project_name> -c <credentials_path>
```
User must provide the project name using flag (-p) and the local path to the JSON file with the credentials (-c).

To test GCP BigQuery there is a script located in <base_path_to_model_zoo>/data_connector/data_connector/samples/gcp/bigquery.py. To run it execute from the base path of data connector for OAuth:
```bash
python -m samples.gcp.bigquery -o
```
For service account:
```bash
python -m samples.gcp.bigquery -p <project_name> -c <credentials_path>
```
User must provide the project name using flag (-p) and the local path to the JSON file with the credentials (-c).
