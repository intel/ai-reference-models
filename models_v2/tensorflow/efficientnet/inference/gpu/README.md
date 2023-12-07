# EfficientNet Model Inference

EfficientNet Model Inference using Intel® Extension for TensorFlow.

## Model Information

| **Use Case** | **Framework** | **Model Repo** | **Branch/Commit/Tag** | **Optional Patch** |
|:---:| :---: |:--------------:|:---------------------:|:------------------:|
|  Inference   |  TensorFlow   |       -        |           -           |         -          |

# Pre-Requisite

* Host has Intel® Data Center GPU FLEX
* Host has installed latest Intel® Data Center GPU Flex Series
  Driver https://dgpu-docs.intel.com/driver/installation.html
* Install [Intel® Extension for TensorFlow](https://pypi.org/project/intel-extension-for-tensorflow/)

1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models/image_recognition/tensorflow/efficientnet/inference/gpu_version2`
 Run `setup.sh` this will create virtual environment `venv`.
   Setup required environment paramaters
 
| **Parameter**             |                                  **export command**                                  |
|:---------------------------:|:------------------------------------------------------------------------------------:|
| **MODEL_NAME**          | `export MODEL_NAME=EfficientNetB0` (EfficientNetB0, EfficientNetB3 or EfficientNetB4) |
| **BATCH_SIZE** (optional) |                               `export BATCH_SIZE=128`                                |

## Output

Output will typicall looks like:

```
load data ......
input shape (128, 224, 224, 3)
Creating model finished.
Batchsize is 128
Avg time: 0.0484589417775472 s.
Throughput: 2641.4113743463354 img/s.
```

Final results of the training run can be found in `results.yaml` file.

```
results:
 - key: throughput
   value: 2641.4113743463354
   unit: img/s
```
