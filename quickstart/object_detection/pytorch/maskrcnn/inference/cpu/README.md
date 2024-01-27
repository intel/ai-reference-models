# Mask R-CNN Inference

## Description
This document has instructions for running Mask R-CNN inference.

## Quick Start Scripts
| Script name | Description |
|-------------|-------------|
| `inference_realtime.sh` | Runs multi instance realtime inference using 4 cores per instance for the specified precision (fp32, avx-fp32, bf16, or bf32) and mode (imperative or jit). |
| `inference_throughput.sh` | Runs multi instance batch inference using 24 cores per instance for the specified precision (fp32, avx-fp32, bf16, or bf32) and mode (imperative or jit). |
| `accuracy.sh` | Measures the inference accuracy for the specified precision (fp32, avx-fp32, bf16, or bf32) and mode (imperative or jit). |

> Note: The `avx-fp32` precisions run the same scripts as `fp32`, except that the
> `DNNL_MAX_CPU_ISA` environment variable is unset. The environment variable is
> otherwise set to `DNNL_MAX_CPU_ISA=AVX512_CORE_AMX`.
* Set ENV to use AMX:
  ```
  export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
  ```

## Bare Metal
### General setup

Follow [link](/docs/general/pytorch/BareMetalSetup.md) to install Miniconda and build Pytorch, IPEX, TorchVison and Jemalloc and TCmalloc

### Model Specific Setup
* Set Jemalloc and tcmalloc Preload for better performance

  The jemalloc should be built from the [General setup](#general-setup) section.
  ```
  export LD_PRELOAD="<path to the jemalloc directory>/lib/libjemalloc.so":"path_to/tcmalloc/lib/libtcmalloc.so":$LD_PRELOAD
  export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
  ```

* Set IOMP preload for better performance

  IOMP should be installed in your conda env from the [General setup](#general-setup) section.
  ```
  pip install packaging intel-openmp
  export LD_PRELOAD=<path to the intel-openmp directory>/lib/libiomp5.so:$LD_PRELOAD
  ```

* Follow the instructions to setup your bare metal environment on either Linux or Windows systems. Once all the setup is done,
  the Intel® AI Reference Models can be used to run a [quickstart script](#quick-start-scripts).
  Ensure that you have a clone of the [Intel® AI Reference Models Github repository](https://github.com/IntelAI/models) and navigate to the directory.
  ```
  git clone https://github.com/IntelAI/models.git
  cd models
  ```

* Install model
  ```
  python models/object_detection/pytorch/maskrcnn/maskrcnn-benchmark/setup.py develop
  ```

* Download pretrained model
  ```
  export CHECKPOINT_DIR=<directory where the pretrained model will be saved>
  bash quickstart/object_detection/pytorch/maskrcnn/inference/cpu/download_model.sh
  ```
* Datasets
  Download the 2017 [COCO dataset](https://cocodataset.org) using the `download_dataset.sh` script.
  Export the `DATASET_DIR` environment variable to specify the directory where the dataset
  will be downloaded. This environment variable will be used again when running quickstart scripts.
  ```
  cd quickstart/object_detection/pytorch/maskrcnn/inference/cpu
  export DATASET_DIR=<directory where the dataset will be saved>
  bash download_dataset.sh
  cd -
  ```

## Run on Linux
  ```
  # Navigate to Intel® AI Reference Models dir:
  cd models

  # Install dependency:
  ./quickstart/object_detection/pytorch/maskrcnn/inference/cpu/setup.sh
  
  # Set environment variables
  export DATASET_DIR=<path to the COCO dataset>
  export CHECKPOINT_DIR=<path to the downloaded pretrained model>
  export OUTPUT_DIR=<path to an output directory>
  export MODE=<set to 'jit' or 'imperative'>
  export PRECISION=< select from :- fp32, avx-fp32, bf16, or bf32>

  # Optional environemnt variables:
  export BATCH_SIZE=<set a value for batch size, else it will run with default batch size>

  # Run a quickstart script (for example, FP32 batch inference jit)
  ./quickstart/object_detection/pytorch/maskrcnn/inference/cpu/<script.sh>
  ```

## Run on Windows
If not already setup, please follow instructions for [environment setup on Windows](/docs/general/Windows.md).

* Install dependencies
  ```
  pip install yacs opencv-python pycocotools defusedxml cityscapesscripts
  conda install intel-openmp
  ```

* Using Windows CMD.exe, run:
  ```
  cd models

  # Env vars
  set DATASET_DIR=<path to the COCO dataset>
  set CHECKPOINT_DIR=<path to the downloaded pretrained model>
  set OUTPUT_DIR=<path to the directory where log files will be written>
  set MODE=<set to 'jit' or 'imperative'>
  set PRECISION=<set to fp32, avx-fp32, bf16, or bf32>

  #Run a quickstart script:
  bash quickstart\object_detection\pytorch\maskrcnn\inference\cpu\<script.sh>
  ```


<!--- 80. License -->
## License

[LICENSE](/LICENSE)
