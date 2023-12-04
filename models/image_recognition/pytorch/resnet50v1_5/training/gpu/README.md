# Resnet50 Training

Resnet50 Training BKC.

## Model Information

| **Use Case** | **Framework** | **Model Repo** | **Branch/Commit/Tag** | **Optional Patch** |
|:---:| :---: |:--------------:|:---------------------:|:------------------:|
|  Training    |    Pytorch    |       -        |           -           |         -          |

# Pre-Requisite
* Host has Intel® Data Center GPU MAX or FLEX
* Host has installed latest Intel® Data Center GPU Max & Flex Series Drivers https://dgpu-docs.intel.com/driver/installation.html

# Prepare Dataset
## Dataset: imagenet
ImageNet is recommended, the download link is https://image-net.org/challenges/LSVRC/2012/2012-downloads.php.

## Training
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models/image_recognition/pytorch/resnet50v1_5/training/gpu`
3. Run `setup.sh` this will install all the required dependencies & create virtual environment `venv`.
4. Activate virtual env: `. ./venv/bin/activate`
5. Setup required environment paramaters

| **Parameter**                |                                  **export command**                                  |
|:---------------------------:|:------------------------------------------------------------------------------------:|
| **PRECISION**                | `export PRECISION=bf16` (bf16 or tf32 or bf32)                                       |
| **MULTI_TILE**               | `export MULTI_TILE=False` (True or False)                                            |
| **PLATFORM**                 | `export PLATFORM=PVC` (PVC or ATS-M)                                                 |
| **DATASET_DIR**              |                               `export DATASET_DIR=`                                  |
| **BATCH_SIZE** (optional)    |                               `export BATCH_SIZE=256`                                |
|**NUM_ITERATIONS** (optional) |                               `export NUM_ITERATIONS=20`                             |
| **OUTPUT_DIR** (optional)    |                               `export OUTPUT_DIR=$PWD`                               |
6. Run `run_model.sh`

## Output

Single-tile output will typicall looks like:

```
Training performance: batch size:256, throughput:1681.75 image/sec
```

Multi-tile output will typicall looks like:
```
[0] Training performance: batch size:256, throughput:1584.89 image/sec
[1] Training performance: batch size:256, throughput:1584.41 image/sec
```
Final results of the training run can be found in `results.yaml` file.

```
results:
 - key: throughput
   value: 1681.75
   unit: it/s
 - key: latency
   value: 0.92756983948693794
   unit: s
 - key: accuracy
   value: 7.6683e+00
   unit: top1
```
