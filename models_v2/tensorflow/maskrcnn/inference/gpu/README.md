# Mask RCNN Inference

Mask RCNN Inference using Intel® Extension for TensorFlow.

## Model Information

| **Use Case** | **Framework** | **Model Repo** | **Branch/Commit/Tag** | **Weight** | **Optional Patch** |
| :---: | :---: | :---: | :---: | :---: | :---: |
|   Inference   |  Tensorflow   | [DeepLearningExamples/MaskRCNN](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Segmentation/MaskRCNN) |        master         | See Section [Prerequisites](#weight) | [EnableInference.patch](#inference patch) |

**Note**: Refer to [CONTAINER.md](CONTAINER.md) for MaskRCNN Inference instructions using docker containers.

# Pre-Requisite
* Host has Intel® Data Center GPU Flex Series
* Host has installed latest Intel® Data Center GPU Flex Series Drivers https://dgpu-docs.intel.com/driver/installation.html
* Install [Intel® Extension for TensorFlow](https://pypi.org/project/intel-extension-for-tensorflow/)
* The following Intel® oneAPI Base Toolkit components are required:
  - Intel® oneAPI DPC++ Compiler (Placeholder DPCPPROOT as its installation path)
  - Intel® oneAPI Math Kernel Library (oneMKL) (Placeholder MKLROOT as its installation path)
  - Intel® oneAPI MPI Library
  - Intel® oneAPI TBB Library
  - Intel® oneAPI CCL Library

  Follow instructions at [Intel® oneAPI Base Toolkit Download page](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?operatingsystem=linux) to setup the package manager repository.

# Dataset 
Download & preprocess COCO 2017 dataset. 
```
export DATASET_DIR=/path/to/dataset/dir
git clone https://github.com/NVIDIA/DeepLearningExamples.git
cd ./DeepLearningExamples/TensorFlow2/Segmentation/MaskRCNN/dataset
bash download_and_preprocess_coco.sh $DATASET_DIR
```

## Inference
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models_v2/tensorflow/maskrcnn/inference/gpu`
3. Create virtual environment `venv` and activate it:
    ```
    python3 -m venv venv
    . ./venv/bin/activate
    ```
4. Run setup.sh
    ```
    ./setup.sh
    ```
5. Install [tensorflow and ITEX](https://pypi.org/project/intel-extension-for-tensorflow/)
6. Download weights
    ```
        pushd .
        cd ./DeepLearningExamples/TensorFlow2/Segmentation/MaskRCNN
        python scripts/download_weights.py --save_dir=./weights
        popd 
    ```
7. Set environment variables for Intel® oneAPI Base Toolkit: 
    Default installation location `{ONEAPI_ROOT}` is `/opt/intel/oneapi` for root account, `${HOME}/intel/oneapi` for other accounts
    ```bash
    source {ONEAPI_ROOT}/compiler/latest/env/vars.sh
    source {ONEAPI_ROOT}/mkl/latest/env/vars.sh
    source {ONEAPI_ROOT}/tbb/latest/env/vars.sh
    source {ONEAPI_ROOT}/mpi/latest/env/vars.sh
    source {ONEAPI_ROOT}/ccl/latest/env/vars.sh
8. Setup required environment paramaters

    |   **Parameter**    | **export command**                                    |
    | :---: | :--- |
    |  **DATASET_DIR**   | `export DATASET_DIR=/the/path/to/dataset`             |
    |   **BATCH_SIZE** (optional)   | `export BATCH_SIZE=4`           |
    |   **PRECISION**   | `export PRECISION=bfloat16` (float16 or fp32)           |
    |   **GPU_TYPE**    | `export GPU_TYPE=<flex_140 or flex_170>`                 |
7. Run `run_model.sh`

## Output

Output typically looks like:
```
2023-09-11 14:54:49,905 I dllogger        (1, 20) loss: 639.5632934570312
2023-09-11 14:54:49,906 I dllogger        (1, 20) train_time: 23.89216899871826, train_throughput: 21.438093303125907
2023-09-11 14:54:49,914 I dllogger        (1,) loss: 639.5632934570312
2023-09-11 14:54:49,914 I dllogger        () loss: 639.5632934570312
2023-09-11 14:54:49,915 I dllogger        () train_time: 23.90118169784546, train_throughput: 23.507529269636105
```

Final results of the training run can be found in `results.yaml` file.
```
results:
 - key: throughput
   value: 32.896633477338334
   unit: images/sec
```
