# SSD-ResNet34 Inference

## Description
This document has instructions for running SSD-ResNet34 Inference using Intel-optimized PyTorch.

## Bare Metal
### General setup

Follow [link](/docs/general/pytorch/BareMetalSetup.md) to install Conda and build Pytorch, IPEX, TorchVison and Jemalloc.

### Model Specific Setup
* Install dependencies
  ```
  pip install matplotlib Pillow pycocotools
  ```

* Download pretrained model
  ```
  cd <path to your clone of the model zoo>/quickstart/object_detection/pytorch/ssd-resnet34/inference/cpu
  export CHECKPOINT_DIR=<directory where to save the pretrained model>
  bash download_model.sh
  ```

* Set Jemalloc Preload for better performance

  The jemalloc should be built from the [General setup](#general-setup) section.
  ```
  export LD_PRELOAD="path/lib/libjemalloc.so":$LD_PRELOAD
  export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
  ```

* Set IOMP preload for better performance

  IOMP should be installed in your conda env from the [General setup](#general-setup) section.
  ```
  export LD_PRELOAD=path/lib/libiomp5.so:$LD_PRELOAD
  ```

* Set ENV to use AMX if you are using SPR
  ```
  export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
  ```

## Datasets

Download the 2017 [COCO dataset](https://cocodataset.org) using the `download_dataset.sh` script.
Export the `DATASET_DIR` environment variable to specify the directory where the dataset
will be downloaded. This environment variable will be used again when running quickstart scripts.
```
cd <path to your clone of the model zoo>/quickstart/object_detection/pytorch/ssd-resnet34/inference/cpu
export DATASET_DIR=<directory where the dataset will be saved>
bash download_dataset.sh
```

## Quick Start Scripts

|  DataType   | Throughput  |  Latency    |   Accuracy  |
| ----------- | ----------- | ----------- | ----------- |
| FP32        | bash inference_throughput.sh fp32 | bash inference_realtime.sh fp32 | bash accuracy.sh fp32 |
| BF16        | bash inference_throughput.sh bf16 | bash inference_realtime.sh bf16 | bash accuracy.sh bf16 |
| INT8        | bash inference_throughput.sh int8 | bash inference_realtime.sh int8 | bash accuracy.sh int8 |

To do the int8 calibration `bash bare_metal_int8_calibration.sh int8 <file where to save the calibrated model> <steps to run calibration>`, for example `bash bare_metal_int8_calibration.sh int8 test.json 100`.

## Run the model

Follow the instructions above to setup your bare metal environment, download and
preprocess the dataset, and do the model specific setup. Once all the setup is done,
the Model Zoo can be used to run a [quickstart script](#quick-start-scripts).
Ensure that you have an enviornment variables set to point to the dataset directory,
the downloaded pretrained model, and an output directory.

```
# Clone the model zoo repo and set the MODEL_DIR
git clone https://github.com/IntelAI/models.git
cd models
export MODEL_DIR=$(pwd)

# Env vars
export DATASET_DIR=<path to the COCO dataset>
export CHECKPOINT_DIR=<path to the pretrained model>
export OUTPUT_DIR=<path to an output directory>

# Run a quickstart script (for example, FP32 batch inference)
cd ${MODEL_DIR}/quickstart/object_detection/pytorch/ssd-resnet34/inference/cpu
bash inference_throughput.sh fp32
```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)