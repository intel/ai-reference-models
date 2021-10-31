<!--- 20. Datasets -->
## Datasets

The dataset used in these containers is [Criteo Terabyte Dataset](https://labs.criteo.com/2013/12/download-terabyte-click-logs/).

To prepare the dataset please for the steps described below.
(1) Go to the Criteo Terabyte Dataset website(https://labs.criteo.com/2013/12/download-terabyte-click-logs/) and accept the terms of use. 
(2) Copy the data download URL in the following page, and run :
```
    mkdir <dir/to/save/dlrm_data> && cd <dir/to/save/dlrm_data>
    curl -O <download url>/day_{$(seq -s , 0 23)}.gz
    gunzip day_*.gz
```
(2) Please remember to replace  "<dir/to/save/dlrm_data>" to any path you want to download and save the dataset.

The folder that contains the "Terabyte Click Logs" dataset should be set as the
`DATASET_DIR` when running quickstart scripts 
(for example: `export DATASET_DIR=/home/<user>/terabyte_dataset`).
