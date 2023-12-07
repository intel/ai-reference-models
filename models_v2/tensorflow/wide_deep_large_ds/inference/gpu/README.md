# Wide and Deep Large Model Inference

Wide and Deep Large Model Inference Intel® Extension for TensorFlow.

## Model Information

| **Use Case** | **Framework** | **Model Repo** | **Branch/Commit/Tag** | **Optional Patch** |
|:---:| :---: |:--------------:|:---------------------:|:------------------:|
|  Inference   |  TensorFlow   |       -        |           -           |         -          |

# Pre-Requisite

* Host has Intel® Data Center GPU FLEX
* Host has installed latest Intel® Data Center GPU Flex Series
  Driver https://dgpu-docs.intel.com/driver/installation.html
# Dataset and pretrained model
* Follow [instructions](https://github.com/IntelAI/models/tree/master/datasets/large_kaggle_advertising_challenge/README.md) to download and preprocess the Large Kaggle Display Advertising Challenge Dataset.
* Get pretrained model: `wget https://storage.googleapis.com/intel-optimized-tensorflow/models/3_0/wide_deep_fp16_pretrained_model.pb`
* Install [Intel® Extension for TensorFlow](https://pypi.org/project/intel-extension-for-tensorflow/)

## Inference

1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models_v2/tensorflow/wide_deep_large_ds/inference/gpu`
 Run `setup.sh` this will create virtual environment `venv`.
3. Setup required environment paramaters
 
|      **Parameter**       |                      **export command**                       |
|:------------------------:|:-------------------------------------------------------------:|
|     **DATASET_PATH**     |          `export DATASET_PATH=/the/path/to/dataset`           |
|     **PB_FILE_PATH**     |             `export PB_FILE_PATH=/the/path/to/pb`             |
| **BATCH_SIZE**(optional) |          `export BATCH_SIZE=10000`                            |

## Output

Output will typicall looks like:

```
--------------------------------------------------
Total test records           :  20000000
Batch size is                :  10000
Number of batches            :  2000
Inference duration (seconds) :  7.5929
Average Latency (ms/batch)   :  3.9963
Throughput is (records/sec)  :  2502326.933
--------------------------------------------------
```

Final results of the training run can be found in `results.yaml` file.

```
results:
 - key: throughput
   value: 2502326.933
   unit: records/sec
```
