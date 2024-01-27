# SSD-MobilenetV1 Inference

SSD-MobileNetV1 Inference best known configurations with Intel® Extension for PyTorch.

## Model Information

| **Use Case** | **Framework** | **Model Repo** | **Branch/Commit/Tag** | **Optional Patch** |
|:---:| :---: |:--------------:|:---------------------:|:------------------:|
|  Inference   |    PyTorch    |       https://github.com/qfgaohao/pytorch-ssd        |           -           |         -          |

# Pre-Requisite
* Host has Intel® Data Center GPU Arc
* Host has installed latest Intel® Data Center GPU Arc Series Drivers https://dgpu-docs.intel.com/driver/installation.html
* The following Intel® oneAPI Base Toolkit components are required:
  - Intel® oneAPI DPC++ Compiler (Placeholder DPCPPROOT as its installation path)
  - Intel® oneAPI Math Kernel Library (oneMKL) (Placeholder MKLROOT as its installation path)
  - Intel® oneAPI MPI Library
  - Intel® oneAPI TBB Library

  Follow instructions at [Intel® oneAPI Base Toolkit Download page](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?operatingsystem=linux) to setup the package manager repository.

# Prepare Dataset
Download, and extract the following dataset, checkpoint and VOC model labels:
# Dataset: VOC2007
Download from URL: [VOCtrainval_06-Nov-2007.tar](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar) and extract, you will find the extract folder as below:
```
VOCDevkit/
└── VOC2007
    ├── Annotations
    ├── ImageSets
    ├── JPEGImages
    ├── SegmentationClass
    └── SegmentationObject
```
please indicator the `DATASET_DIR` to path/to/your/folder/VOCdevkit/VOC2007

# Pretrained model
Download form [this URL](https://drive.google.com/drive/folders/1pKn-RifvJGWiOx0ZCRLtCXM5GT5lAluu?usp=sharing) to get mobilenet-v1-ssd-mp-0_675.pth and put under `WEIGHT_DIR`

# Voc-model-labels
Download from [this URL](https://drive.google.com/drive/folders/1pKn-RifvJGWiOx0ZCRLtCXM5GT5lAluu?usp=sharing) to get voc-model-labels.txt and put under `LABEL_DIR`


## Inference
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models_v2/pytorch/ssd-mobilenetv1/inference/gpu`
3. Create virtual environment `venv` and activate it:
    ```
    python3 -m venv venv
    . ./venv/bin/activate
    ```
4. Run setup.sh
    ```
    ./setup.sh
    ```
5. Install the latest GPU versions of [torch, torchvision and intel_extension_for_pytorch](https://intel.github.io/intel-extension-for-pytorch/index.html#installation):
  ```
  python -m pip install torch==<torch_version> torchvision==<torchvvision_version> intel-extension-for-pytorch==<ipex_version> --extra-index-url https://pytorch-extension.intel.com/release-whl-aitools/
  ```
6. Set environment variables for Intel® oneAPI Base Toolkit: 
    Default installation location `{ONEAPI_ROOT}` is `/opt/intel/oneapi` for root account, `${HOME}/intel/oneapi` for other accounts
    ```bash
    source {ONEAPI_ROOT}/compiler/latest/env/vars.sh
    source {ONEAPI_ROOT}/mkl/latest/env/vars.sh
    source {ONEAPI_ROOT}/tbb/latest/env/vars.sh
    source {ONEAPI_ROOT}/mpi/latest/env/vars.sh
    source {ONEAPI_ROOT}/ccl/latest/env/vars.sh
    ```
7. Setup required environment paramaters

| **Parameter**                |                                  **export command**                                  |
|:---------------------------:|:------------------------------------------------------------------------------------:|
| **MULTI_TILE**               | `export MULTI_TILE=False` (False)                                             |
| **PLATFORM**                 | `export PLATFORM=ARC` (ARC)                                                   |
| **DATASET_DIR**              |                               `export DATASET_DIR=` (the dir path to voc2007 in dataset)|
| **WEIGHT_DIR**              |                               `export WEIGHT_DIR=` (contain .pth files)                             |
| **LABEL_DIR**              |                               `export LABEL_DIR=` (contain .txt files)                            |
| **BATCH_SIZE** (optional)    |                               `export BATCH_SIZE=256`                                |
| **PRECISION**      |                               `export PRECISION=INT8` (INT8, FP16 and FP32 for ARC)  |
| **OUTPUT_DIR** (optional)    |                               `export OUTPUT_DIR=$PWD`                               |
|**NUM_ITERATIONS** (optional) |                               `export NUM_ITERATIONS=500`                             |
| **DATASET_DIR** (optional)   |                               `export DATASET_DIR=--dummy`                           |
8. Run `run_model.sh`

## Output

Single-tile output will typically looks like:

```
process image 500
Load Image: 0.000002 seconds.
Inference time:  0.02182316780090332
Post time:  0.003971099853515625
Prediction: 0.438544 seconds.
```

```
Final results of the inference run can be found in `results.yaml` file.
```
results:
 - key: throughput
   value: 11721.356
   unit: inst/s
 - key: latency
   value: 0.021840476989746095
   unit: s
 - key: accuracy
   value: None
   unit: accuracy
```
