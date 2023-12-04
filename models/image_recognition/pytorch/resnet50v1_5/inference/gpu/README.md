# Resnet50 Inference

Resnet50 Inference BKC.

## Model Information

| **Use Case** | **Framework** | **Model Repo** | **Branch/Commit/Tag** | **Optional Patch** |
|:---:| :---: |:--------------:|:---------------------:|:------------------:|
|  Inference   |    Pytorch    |       -        |           -           |         -          |

# Pre-Requisite
* Host has Intel® Data Center GPU MAX or FLEX
* Host has installed latest Intel® Data Center GPU Max & Flex Series Drivers https://dgpu-docs.intel.com/driver/installation.html

# Prepare Dataset
## Dataset: imagenet
Default is dummy dataset.
ImageNet is recommended, the download link is https://image-net.org/challenges/LSVRC/2012/2012-downloads.php.


## Inference
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models/image_recognition/pytorch/resnet50v1_5/inference/gpu`
3. Run `setup.sh` this will install all the required dependencies & create virtual environment `venv`.
4. Activate virtual env: `. ./venv/bin/activate`
5. Setup required environment paramaters

| **Parameter**                |                                  **export command**                                  |
|:---------------------------:|:------------------------------------------------------------------------------------:|
| **MULTI_TILE**               | `export MULTI_TILE=True` (True or False)                                             |
| **PLATFORM**                 | `export PLATFORM=PVC` (PVC or ATS-M)                                                 |
| **BATCH_SIZE** (optional)    |                               `export BATCH_SIZE=1024`                                |
| **PRECISION** (optional)     |                               `export PRECISION=int8`                                |
| **OUTPUT_DIR** (optional)    |                               `export OUTPUT_DIR=$PWD`                               |
|**NUM_ITERATIONS** (optional) |                               `export NUM_ITERATIONS=500`                             |
| **DATASET_DIR** (optional)   |                               `export DATASET_DIR=--dummy`                           |
6. Run `run_model.sh`

## Output

Single-tile output will typicall looks like:

```
Quantization Evalution performance: batch size:256, throughput:26372.99 image/sec, Acc@1:0.10, Acc@5:0.45
```

Multi-tile output will typicall looks like:
```
Quantization Evalution performance: batch size:1024, throughput:26372.99 image/sec, Acc@1:0.10, Acc@5:0.5
Quantization Evalution performance: batch size:1024, throughput:24552.76 image/sec, Acc@1:0.10, Acc@5:0.5
```

Final results of the inference run can be found in `results.yaml` file.
```
results:
 - key: throughput
   value: 26372.99
   unit: it/s
 - key: latency
   value: 0.02754672721207752
   unit: s
 - key: accuracy
   value: 0.10
   unit: top1
```
