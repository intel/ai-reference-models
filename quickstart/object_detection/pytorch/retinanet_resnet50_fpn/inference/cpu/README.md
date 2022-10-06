# RetinaNet ResNet-50 FPN Inference

## Description
This document has instructions for running RetinaNet ResNet-50 FPN Inference.

## Datasets

Download the 2017 [COCO dataset](https://cocodataset.org) using the `download_dataset.sh` script.
Export the `DATASET_DIR` environment variable to specify the directory where the dataset
will be downloaded. This environment variable will be used again when running quickstart scripts.
```
cd <path to your clone of the model zoo>/quickstart/object_detection/pytorch/retinanet_resnet50_fpn/inference/cpu
export DATASET_DIR=<directory where the dataset will be saved>
bash download_dataset.sh
```

## Quick Start Scripts

|  DataType   | Throughput  |  Latency    |   Accuracy  |
| ----------- | ----------- | ----------- | ----------- |
| FP32        | bash batch_inference_baremetal.sh fp32 | bash online_inference_baremetal.sh fp32 | bash accuracy_baremetal.sh fp32 |
| BF16        | bash batch_inference_baremetal.sh bf16 | bash online_inference_baremetal.sh bf16 | bash accuracy_baremetal.sh bf16 |

Follow the instructions to setup your bare metal environment on either Linux or Windows systems. Once all the setup is done,
the Model Zoo can be used to run a [quickstart script](#quick-start-scripts).
Ensure that you have a clone of the [Model Zoo Github repository](https://github.com/IntelAI/models).
```
git clone https://github.com/IntelAI/models.git
```

## Run on Linux

Follow [link](/docs/general/pytorch/BareMetalSetup.md) to install Conda and build Pytorch, IPEX, TorchVison and Jemalloc.

* Install dependencies
  ```
  pip install Pillow pycocotools
  ```

* Set Jemalloc Preload for better performance

  After [Jemalloc setup](/docs/general/pytorch/BareMetalSetup.md#build-jemalloc), set the following environment variables.
  ```
  export LD_PRELOAD="<path to the jemalloc directory>/lib/libjemalloc.so":$LD_PRELOAD
  export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
  ```

* Set IOMP preload for better performance

  IOMP should be installed in your conda env. Set the following environment variables.
  ```
  export LD_PRELOAD=<path to the intel-openmp directory>/lib/libiomp5.so:$LD_PRELOAD
  ```

* Set ENV to use AMX if you are using SPR
  ```
  export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
  ```

* Run the model:
  ```
  cd models

  # Set environment variables
  export DATASET_DIR=<path to the COCO dataset>
  export OUTPUT_DIR=<path to an output directory>

  # Run a quickstart script (for example, FP32 batch inference)
  bash quickstart/object_detection/pytorch/retinanet_resnet50_fpn/inference/cpu/batch_inference_baremetal.sh fp32
  ```

## Run on Windows
If not already setup, please follow instructions for [environment setup on Windows](/docs/general/Windows.md).

* Install dependencies
  ```
  pip install Pillow pycocotools
  ```

* Using Windows CMD.exe, run:
  ```
  cd models

  # Env vars
  set DATASET_DIR=<path to the COCO dataset>
  set OUTPUT_DIR=<path to the directory where log files will be written>

  #Run a quickstart script for fp32 precision(FP32 online inference or batch inference or accuracy)
  bash quickstart\object_detection\pytorch\retinanet_resnet50_fpn\inference\cpu\batch_inference_baremetal.sh fp32
  ```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)