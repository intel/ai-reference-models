# SSD-ResNet34 Inference

## Description
This document has instructions for running SSD-ResNet34 Inference using Intel-optimized PyTorch.

## Quick Start Scripts
| Script name | Description |
|-------------|-------------|
| `inference_realtime.sh` | Runs multi instance realtime inference using 4 cores per instance for the specified precision (fp32, avx-fp32, int8, avx-int8, bf32 or bf16). |
| `inference_throughput.sh` | Runs multi instance batch inference using 1 instance per socket for the specified precision (fp32, avx-fp32, int8, avx-int8, bf32 or bf16). |
| `accuracy.sh` | Measures the inference accuracy (providing a `DATASET_DIR` environment variable is required) for the specified precision (fp32, avx-fp32, int8, avx-int8, bf32 or bf16). |

**Note:** The `avx-int8` and `avx-fp32` precisions run the same scripts as `int8` and `fp32`, except that the `DNNL_MAX_CPU_ISA` environment variable is unset. The environment variable is otherwise set to `DNNL_MAX_CPU_ISA=AVX512_CORE_AMX`
* Set ENV to use AMX:
  ```
  export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
  ```

To do the int8 calibration bash bare_metal_int8_calibration.sh int8 <file where to save the calibrated model> <steps to run calibration>, for example bash bare_metal_int8_calibration.sh int8 test.json 10.

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

## Run the model
Once all the setup is done, the Intel® AI Reference Models can be used to run a [quickstart script](#quick-start-scripts).
Ensure that you have an enviornment variables set to point to the dataset directory,
the downloaded pretrained model, and an output directory.

```
# Clone the Intel® AI Reference Models repo and set the MODEL_DIR
git clone https://github.com/IntelAI/models.git
cd models
export MODEL_DIR=$(pwd)

# Dataset
Download the 2017 [COCO dataset](https://cocodataset.org) using the `download_dataset.sh` script.
Export the `DATASET_DIR` environment variable to specify the directory where the dataset
will be downloaded. This environment variable will be used again when running quickstart scripts.

cd quickstart/object_detection/pytorch/ssd-resnet34/inference/cpu
export DATASET_DIR=<directory where the dataset will be saved>
bash download_dataset.sh
cd - 
cd ${MODEL_DIR}

# install model specific dependencies
./quickstart/object_detection/pytorch/ssd-resnet34/inference/cpu/setup.sh

# Download pretrained model
export CHECKPOINT_DIR=<directory where to save the pretrained model>
bash quickstart/object_detection/pytorch/ssd-resnet34/inference/cpu/download_model.sh

# Env vars
export DATASET_DIR=<path to the COCO dataset>
export OUTPUT_DIR=<path to an output directory>
export PRECISION=<select from :- int8, avx-int8, fp32, avx-fp32, bf16, or bf32>

# Optional environemnt variables:
export BATCH_SIZE=<set a value for batch size, else it will run with default batch size>

# Run a quickstart script:
./quickstart/object_detection/pytorch/ssd-resnet34/inference/cpu/<quickstart_script.sh>
```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)