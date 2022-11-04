# TransNetV2 Inference

## Description
This document has instructions for running TransNetV2 Inference.

## Quick Start Scripts

|  DataType   | Throughput  |  Latency    |
| ----------- | ----------- | ----------- |
| FP32        | bash inference_throughput.sh fp32 | bash inference_realtime.sh fp32 |
| BF16        | bash inference_throughput.sh bf16 | bash inference_realtime.sh bf16 |

Follow the instructions to setup your bare metal environment on either Linux or Windows systems. Once all the setup is done,
the Model Zoo can be used to run a [quickstart script](#quick-start-scripts).
Ensure that you have a clone of the [Model Zoo Github repository](https://github.com/IntelAI/models).
```
git clone https://github.com/IntelAI/models.git
```

## Run the model on Linux

Follow [link](/docs/general/pytorch/BareMetalSetup.md) to install Miniconda and build Pytorch, IPEX, TorchVison and Jemalloc.

* Install dependencies
  ```
  pip install ffmpeg-python matplotlib Pillow pycocotools pandas
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
  export OUTPUT_DIR=<path to an output directory>

  # Run a quickstart script (for example, FP32 batch inference)
  bash quickstart/shot_boundary_detection/pytorch/transnetv2/inference/cpu/inference_throughput.sh fp32
  ```

## Run the model on Windows
If not already setup, please follow instructions for [environment setup on Windows](/docs/general/Windows.md).
* Install dependencies
  ```
  pip install ffmpeg-python matplotlib Pillow pycocotools pandas
  ```
* Using Windows CMD.exe, run:
  ```
  cd models

  # Set environment variables
  set OUTPUT_DIR=<path to an output directory>

  # Run a quickstart script (FP32 realtime inference or batch inference)
  bash quickstart\shot_boundary_detection\pytorch\transnetv2\inference\cpu\inference_realtime.sh fp32
  ```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)
