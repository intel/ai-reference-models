# DistilBert Inference

DistilBert Inference best known configurations with Intel® Extension for PyTorch.

## Model Information

| **Use Case** | **Framework** | **Model Repo** | **Branch/Commit/Tag** | **Optional Patch** |
|:---:| :---: |:--------------:|:---------------------:|:------------------:|
|  Inference   |    PyTorch    |       https://github.com/huggingface/transformers/tree/main/src/transformers/models/distilbert        |           -           |         -          |

# Pre-Requisite
* Host has Intel® Data Center GPU MAX or FLEX
* Host has installed latest Intel® Data Center GPU Max & Flex Series Drivers https://dgpu-docs.intel.com/driver/installation.html

# Prepare Dataset
## Dataset: 
please refer to https://github.com/mlcommons/training_results_v2.1/tree/main/NVIDIA/benchmarks/bert/implementations/pytorch-22.09#download-and-prepare-the-data

the dataset should be like below
|_hdf5  
      |_ eval                               # evaluation chunks in binary hdf5 format fixed length (not used in training, can delete after data   preparation)  
      |_ eval_varlength                     # evaluation chunks in binary hdf5 format variable length *used for training*
      |_ training                           # 500 chunks in binary hdf5 format 
      |_ training_4320                      # 
      |_ hdf5_4320_shards_uncompressed   # sharded data in hdf5 format fixed length (not used in training, can delete after data   preparation)
      |_ hdf5_4320_shards_varlength      # sharded data in hdf5 format variable length *used for training

## Inference
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models_v2/pytorch/distilbert/inference/gpu`
3. Run `setup.sh` this will install all the required dependencies & create virtual environment `venv`.
4. Activate virtual env: `. ./venv/bin/activate`
5. Setup required environment paramaters

| **Parameter**                |                                  **export command**                                  |
|:---------------------------:|:------------------------------------------------------------------------------------:|
| **MULTI_TILE**               | `export MULTI_TILE=True` (True or False)                                             |
| **PLATFORM**                 | `export PLATFORM=PVC` (PVC or ATS-M)                                                 |
| **DATASET_DIR**              |                               `export DATASET_DIR=`                                  |
| **BATCH_SIZE** (optional)    |                               `export BATCH_SIZE=32`                                |
| **PRECISION** (optional)     | `export PRECISION=BF16` (FP32, BF16, FP16 and TF32 for PVC and FP16, FP32 for ATS-M)|
| **OUTPUT_DIR** (optional)    |                               `export OUTPUT_DIR=$PWD`                               |
|**NUM_ITERATIONS** (optional) |                               `export NUM_ITERATIONS=300`                             |
6. Run `run_model.sh`

## Output

Single-tile output will typicall looks like:

```
12/21/2023 14:28:08 - INFO - utils - PID: 148054 -  --- Ending inference
12/21/2023 14:28:08 - INFO - utils - PID: 148054 -  Results: {'acc': 0.5852613944871974, 'eval_loss': 1.9857747395833334}
12/21/2023 14:28:08 - INFO - utils - PID: 148054 -  The total_time 8.032912015914917 s, and perf 1115.411196120202 sentences/s for inference
12/21/2023 14:28:08 - INFO - utils - PID: 148054 -  Let's go get some drinks.
```

Multi-tile output will typicall looks like:
```
12/21/2023 14:33:13 - INFO - utils - PID: 148381 -  --- Ending inference
12/21/2023 14:33:13 - INFO - utils - PID: 148381 -  Results: {'acc': 0.5852613944871974, 'eval_loss': 1.9857747395833334}
12/21/2023 14:33:13 - INFO - utils - PID: 148381 -  The total_time 8.122166156768799 s, and perf 1103.1539895958633 sentences/s for inference
-Iter:   5%|▍         | 296/6087 [00:12<03:27, 27.93it/s]12/21/2023 14:33:13 - INFO - utils - PID: 148381 -  Let's go get some drinks.
-Iter:   5%|▍         | 300/6087 [00:12<03:56, 24.42it/s]
12/21/2023 14:33:13 - INFO - utils - PID: 148383 -  --- Ending inference
12/21/2023 14:33:13 - INFO - utils - PID: 148383 -  Results: {'acc': 0.5852613944871974, 'eval_loss': 1.9857747395833334}
12/21/2023 14:33:13 - INFO - utils - PID: 148383 -  The total_time 8.266947984695435 s, and perf 1083.834084427241 sentences/s for inference
12/21/2023 14:33:13 - INFO - utils - PID: 148383 -  Let's go get some drinks.
```

Final results of the inference run can be found in `results.yaml` file.
```
results:
 - key: throughput
   value: 2186.9881
   unit: sent/s
 - key: latency
   value: 0.0292663
   unit: s
 - key: accuracy
   value: 0.5850
   unit: acc
```
