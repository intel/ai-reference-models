# Azure ML Sample

Azure ml sample is based on [_Azure Machine Learning in a Day_](https://learn.microsoft.com/en-us/azure/ma)

## Requirements
* Python >= 3.8
* Azure ml account active
* Azure CLI installed
* Azure CLI ML extension installed
* Windows 10 or superior
* An AzureML workspace
* config.json
------

### Install CLI and ML extension

Download latest [Azure CLI](https://aka.ms/installazurecliwindows) installer for windows: 
[Official doc](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli-windows?tabs=azure-cli)

This sample applies __AzureCliCredential__ as default but you can change it setting connection type on connection creator.

#### ML extension 

Once Azure CLI is installed you can install ML extension 
```bash
> az extension add -n ml
```
[Official doc](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli?tabs=public)

-----

#### Config File
As requirement AzureML requires a config json file, you can find a sample file on ./config.json, you should use your own data from AzureML console from your workspace.

You can go to main view on AzureMl to create your first empty workspace to start making samples.