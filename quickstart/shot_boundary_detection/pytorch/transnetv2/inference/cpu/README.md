# TransNetV2 Inference

## Description
This document has instructions for running TransNetV2 Inference using Intel-optimized PyTorch.

## Bare Metal
### General setup

Follow [link](/docs/general/pytorch/BareMetalSetup.md) to install Conda and build Pytorch, IPEX, TorchVison and Jemalloc.

### Model Specific Setup
* Install dependencies
  ```
  pip install ffmpeg-python matplotlib Pillow pycocotools pandas
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

## Quick Start Scripts

|  DataType   | Throughput  |  Latency    |
| ----------- | ----------- | ----------- |
| FP32        | bash inference_throughput.sh fp32 | bash inference_realtime.sh fp32 |
| BF16        | bash inference_throughput.sh bf16 | bash inference_realtime.sh bf16 |


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
export OUTPUT_DIR=<path to an output directory>

# Run a quickstart script (for example, FP32 batch inference)
cd ${MODEL_DIR}/quickstart/shot_boundary_detection/pytorch/transnetv2/inference/cpu
bash inference_throughput.sh fp32
```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)
