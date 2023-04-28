[Main](../../README.md) | [GCP](../../data_connector/gcp/README.md) | [AWS](../../data_connector/aws/README.md) | [Azure](../../data_connector/azure/README.md) | [AzureML](../../data_connector/azure/AzureML.md)

# Cloud interconnection

> This sample connect services between AzureML, Amazon WS and Google CP for Windows PS.

[Interconnection](Interconnection.ipynb)

Our goal is train a machine learning model for credit cards, by now we have all we need in our github repository, we recommend clone data_connector repository and follow all next instructions.

First of all you need an active account on GCP, AWS and AzureML, once you have an account on each you need some requirements to accomplish this sample, these requirements are related with configurations and values you need to connect to the cloud. 
If you want to know more review [GCP Readme](../../data_connector/gcp/README.md), [AWS Readme](../../data_connector/aws/README.md) and [AzureML readme](../../data_connector/azure/README.md)


This sample require you make a _.env_ file and _config.json_, follow reference values from _.emv.sample_ and _config.json.sample_

| Windows Setup: You can run Setup.ps1 to configure basic jupyter notebook to see this sample 

### GCP 
Follow instructions for [GCP Readme](../../data_connector/gcp/README.md), we will work witth environment variables to handle connections, review [.env](.env.sample) file and fill GCP values.

