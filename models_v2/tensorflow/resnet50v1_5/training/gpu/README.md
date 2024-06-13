# ResNet50 Model Training Convergence for Intel® Extention for TensorFlow

## Model Information
| **Case** |**Framework** | **Model Repo** | **Tag** 
| :---: | :---: | :---: | :---: |
| Training | TensorFlow | [TensorFlow-Models](https://github.com/tensorflow/models) | v2.14.0 |

# Pre-Requisite
* Host has Intel® Data Center GPU Max
* Host has installed latest Intel® Data Center GPU Max Series Drivers https://dgpu-docs.intel.com/driver/installation.html
* The following Intel® oneAPI Base Toolkit components are required:
  - Intel® oneAPI DPC++ Compiler (Placeholder DPCPPROOT as its installation path)
  - Intel® oneAPI Math Kernel Library (oneMKL) (Placeholder MKLROOT as its installation path)
  - Intel® oneAPI MPI Library
  - Intel® oneAPI TBB Library
  - Intel® oneAPI CCL Library

  Follow instructions at [Intel® oneAPI Base Toolkit Download page](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?operatingsystem=linux) to setup the package manager repository.

# Dataset 
Using TensorFlow Datasets.
`classifier_trainer.py`` supports ImageNet with [TensorFlow Datasets(TFDS)](https://www.tensorflow.org/datasets/overview) .

Please see the following [example snippet](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/scripts/download_and_prepare.py) for more information on how to use TFDS to download and prepare datasets, and specifically the [TFDS ImageNet readme](https://github.com/tensorflow/datasets/blob/master/docs/catalog/imagenet2012.md) for manual download instructions.

Legacy TFRecords
Download the ImageNet dataset and convert it to TFRecord format. The following [script](https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py) and [README](https://github.com/tensorflow/tpu/tree/master/tools/datasets#imagenet_to_gcspy) provide a few options.

> Note that the legacy ResNet runners, e.g. [resnet/resnet_ctl_imagenet_main.py](https://github.com/tensorflow/models/blob/v2.8.0/official/vision/image_classification/resnet/resnet_ctl_imagenet_main.py) require TFRecords whereas `classifier_trainer.py` can use both by setting the builder to 'records' or 'tfds' in the configurations.

## Training
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models_v2/tensorflow/resnet50v1_5/training/gpu`
3. create virtual environment `venv` and activate it:
    ```
    python3 -m venv venv
    . ./venv/bin/activate
    ```
4. Run setup.sh
    ```
    ./setup.sh
    ```
5. Install [tensorflow and ITEX](https://pypi.org/project/intel-extension-for-tensorflow/)
6. Set environment variables for Intel® oneAPI Base Toolkit: 
    Default installation location `{ONEAPI_ROOT}` is `/opt/intel/oneapi` for root account, `${HOME}/intel/oneapi` for other accounts
    ```bash
    source {ONEAPI_ROOT}/compiler/latest/env/vars.sh
    source {ONEAPI_ROOT}/mkl/latest/env/vars.sh
    source {ONEAPI_ROOT}/tbb/latest/env/vars.sh
    source {ONEAPI_ROOT}/mpi/latest/env/vars.sh
    source {ONEAPI_ROOT}/ccl/latest/env/vars.sh
7. Setup required environment paramaters
   
    |   **Parameter**    | **export command**                                    |
    | :---: | :---: |
    |   **OUTPUT_DIR**   | `export OUTPUT_DIR=/the/path/to/output_dir`           |
    |   **MULTI_TILE**   | `export MULTI_TILE=False` (provide True for multi-tile GPU such as Max 1550, and False for single-tile GPU such as Max 1100)             |
    |  **NUM_DEVICES**   |  `export NUM_DEVICES=<num_devices>` (`<num_devices>` is the number of GPU devices to use for training. It must be equal to or smaller than the number of GPU devices attached to each node. For GPU with 2 tiles, such as Max 1550 GPU, the number of GPU devices in each node is 2 times the number of GPUs, so `<num_devices>` can be set as <=16 for a node with 8 Max 1550 GPUs. While for GPU with single tile, such as Max 1100 GPU, the number of GPU devices available in each node is the same as number of GPUs, so `<num_devices>` can be set as <=8 for a node with 8 Max 1100 GPUs.)     | 
    |   **CONFIG_FILE**   | `export CONFIG_FILE=path/to/itex_xx.yaml` (choose based on NUM_DEVICES used for training, dataset type and precision, see details in the note below)  |
    |  **DATASET_DIR** (optional)  | `export DATASET_DIR=/the/path/to/dataset` (if you choose dummy data, you can ignore this parameter)          |

> [!NOTE]
> Please refer to the below table to set the `CONFIG_FILE`. For single-device training, use one of the yaml file under the `configure` directory while using one of the yaml file under the `hvd_configure` directory for multi-device (NUM_DEVICES>1) distributed training with Horovod.

| **NUM_DEVICES** | **Dataset Type** | **Precision** | **CONFIG FILE**|
| :---: | :---: | :---: | :---: |
|   1   | Dummy | BF16  | `configure/itex_dummy_bf16.yaml`
|   1   | Dummy | FP32  | `configure/itex_dummy_fp32.yaml`
|   1   | Real  | BF16  | `configure/itex_bf16.yaml`
|   1   | Real  | FP32  | `configure/itex_fp32.yaml`
|   >1  | Dummy | BF16  | `hvd_configure/itex_dummy_bf16_lars.yaml`
|   >1  | Dummy | FP32  | `hvd_configure/itex_dummy_fp32_lars.yaml`
|   >1  | Real  | BF16  | `hvd_configure/itex_bf16_lars.yaml`
|   >1  | Real  | FP32  | `hvd_configure/itex_fp32_lars.yaml`

8. Run `run_model.sh`

## Output

Single-device output will typically look like:
```
I1101 12:22:02.439692 139875177744192 keras_utils.py:145] TimeHistory: xxx seconds, xxx examples/second between steps 0 and 100
I1101 12:22:51.165375 139875177744192 keras_utils.py:145] TimeHistory: xxx seconds, xxx examples/second between steps 100 and 200
I1101 12:23:39.856714 139875177744192 keras_utils.py:145] TimeHistory: xxx seconds, xxx examples/second between steps 200 and 300
I1101 12:24:28.548917 139875177744192 keras_utils.py:145] TimeHistory: xxx seconds, xxx examples/second between steps 300 and 400

```

Multi-device output will typically look like:
```
[1] I0607 02:58:54.878461 140172183390016 keras_utils.py:145] TimeHistory: xxx seconds, xxx examples/second between steps 0 and 200
[0] I0607 02:58:54.878722 139770592667456 keras_utils.py:145] TimeHistory: xxx seconds, xxx examples/second between steps 0 and 200
[0] I0607 03:00:17.279742 139770592667456 keras_utils.py:145] TimeHistory: xxx seconds, xxx examples/second between steps 200 and 400
[1] I0607 03:00:17.279656 140172183390016 keras_utils.py:145] TimeHistory: xxx seconds, xxx examples/second between steps 200 and 400
```

Final results of the training run can be found in `results.yaml` file.
```
results:
 - key: throughput
   value: xxx
   unit: images/sec
```
