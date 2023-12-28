# DLRM v1 Inference

DLRM v1 Inference best known configurations with Intel® Extension for PyTorch.

## Model Information

| **Use Case** | **Framework** | **Model Repo** | **Branch/Commit/Tag** | **Optional Patch** |
|:---:| :---: |:--------------:|:---------------------:|:------------------:|
|  Inference   |    PyTorch    |       https://github.com/facebookresearch/dlrm        |           -           |         -          |

# Pre-Requisite
* Host has Intel® Data Center GPU FLex
* Host has installed latest Intel® Data Center GPU Flex Series Drivers https://dgpu-docs.intel.com/driver/installation.html

# Prepare Dataset
The code supports interface with the [Criteo Kaggle Display Advertising Challenge Dataset](https://ailab.criteo.com/ressources/).
   - Please do the following to prepare the dataset for use with DLRM code:
     - First, specify the raw data file (train.txt) as downloaded with
     - This is then pre-processed (categorize, concat across days...) to allow using with dlrm code
     - The processed data is stored as *.npz file
datset dir need have train.txt and kaggleAdDisplayChallenge_processed.npz

you can get the checkpoints by running the command
./bench/dlrm_s_criteo_kaggle.sh [--test-freq=1024]


## Inference
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models_v2/pytorch/dlrm/inference/gpu`
3. Run `setup.sh` this will install all the required dependencies & create virtual environment `venv`.
4. Activate virtual env: `. ./venv/bin/activate`
5. Setup required environment paramaters

| **Parameter**                |                                  **export command**                                  |
|:---------------------------:|:------------------------------------------------------------------------------------:|
| **MULTI_TILE**               | `export MULTI_TILE=False` (False)                                             |
| **PLATFORM**                 | `export PLATFORM=ATS-M` (ATS-M)                                                 |
| **DATASET_DIR**              |                               `export DATASET_DIR=`                                  |
| **CKPT_DIR**                 |                               `export CKPT_DIR=`                                     |
| **BATCH_SIZE** (optional)    |                               `export BATCH_SIZE=32768`                                |
| **PRECISION** (optional)     |                               `export PRECISION=fp16` (fp16 and fp32 forATS-M)                               |
| **OUTPUT_DIR** (optional)    |                               `export OUTPUT_DIR=$PWD`                               |
|**NUM_ITERATIONS** (optional) |                               `export NUM_ITERATIONS=20`                             |
6. Run `run_model.sh`

## Output

Single-tile output will typicall looks like:

```
accuracy 76.215 %, best 76.215 %
dlrm_inf latency:  0.11193203926086426  s
dlrm_inf avg time:  0.007462135950724284  s, ant the time count is : 15
dlrm_inf throughput:  4391235.996821996  samples/s
```


Final results of the inference run can be found in `results.yaml` file.
```
results:
 - key: throughput
   value: 4391236.0
   unit: inst/s
 - key: latency
   value: 0.007462135950724283
   unit: s
 - key: accuracy
   value: 76.215
   unit: accuracy
```
