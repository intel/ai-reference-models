# PyTorch YOLOv7 inference

## Description

This document has instructions for running YOLOv7 inference using
Intel-optimized PyTorch.

## Model Information

| Use Case    | Framework   | Model Repository| Branch/Commit| Patch |
|-------------|-------------|-----------------|--------------|--------------|
| Inference   | Pytorch     | https://github.com/WongKinYiu/yolov7 | main/a207844 | [`yolov7_ipex.patch`](/models/object_detection/pytorch/yolov7/yolov7_ipex.patch). Enable yolov7 inference with IPEX for specified precision (fp32, int8, bf16, fp16, or bf32). |

## Quick Start Scripts

| Script name | Description |
|-------------|-------------| 
| [`inference_realtime.sh`](/quickstart/object_detection/pytorch/yolov7/inference/cpu/inference_realtime.sh) | Runs inference (batch_size=1) for the specified precision (fp32, int8, bf16, fp16, or bf32). |
| [`inference_throughput.sh`](/quickstart/object_detection/pytorch/yolov7/inference/cpu/inference_throughput.sh) | Runs inference to get the throughput for the specified precision (fp32, int8, bf16, fp16, or bf32). |
| [`accuracy.sh`](/quickstart/object_detection/pytorch/yolov7/inference/cpu/accuracy.sh) | Measures the model accuracy (batch_size=40) for the specified precision (fp32, int8, bf16, fp16, or bf32). |

* If you are running fp16 on a platform supporting AMX-FP16, add the following environment variables to leverage AMX-FP16 for better performance.

```
    export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX_FP16
```

* If you are running bf16/fp16/int8 on a platform supporting AVX2-VNNI-2, add the following environment variables to leverage AVX2-VNNI-2 for better performance.

```
    export DNNL_MAX_CPU_ISA=AVX2_VNNI_2
```
For more detailed information about `DNNL_MAX_CPU_ISA` of oneDNN, see oneDNN [link](https://oneapi-src.github.io/oneDNN/index.html).

## Run the Model on Bare Metal

Follow [link](/docs/general/pytorch/BareMetalSetup.md) to install Miniconda and build Pytorch, IPEX, TorchVison and Jemalloc.

* Set Jemalloc Preload for better performance

  After [Jemalloc setup](/docs/general/pytorch/BareMetalSetup.md#build-jemalloc), set the following environment variables.
  ```
  export LD_PRELOAD="<path to the jemalloc directory>/lib/libjemalloc.so":$LD_PRELOAD
  export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
  ```

* Set IOMP preload for better performance

  IOMP should be installed in your conda env. Set the following environment variables.
  ```
  pip install packaging intel-openmp
  export LD_PRELOAD=<path to the intel-openmp directory>/lib/libiomp5.so:$LD_PRELOAD
  ```

* Follow the instructions to setup your bare metal environment on either Linux or Windows systems. Once all the setup is done,
  the Intel® AI Reference Models can be used to run a [quickstart script](#quick-start-scripts).
  Ensure that you have a clone of the [Intel® AI Reference Models Github repository](https://github.com/IntelAI/models) and navigate to models directory.
  ```
  git clone https://github.com/IntelAI/models.git
  cd models
  export MODEL_DIR=$(pwd)
  ```

* Run the model:

    ### Install dependencies
    ```
    pip install pycocotools>=2.0
    ```

    #### Download pretrained model
    Export the `CHECKPOINT_DIR` environment variable to specify the directory where the pretrained model
    will be saved. This environment variable will also be used when running quickstart scripts.
    ```
    cd quickstart/object_detection/pytorch/yolov7/inference/cpu/
    export CHECKPOINT_DIR=<directory where the pretrained model will be saved>
    chmod a+x *.sh
    ./download_model.sh
    ```

    #### Datasets
    Prepare the 2017 [COCO dataset](https://cocodataset.org) for yolov7 using the `download_dataset.sh` script.
    Export the `DATASET_DIR` environment variable to specify the directory where the dataset
    will be saved. This environment variable will also be used when running quickstart scripts.
    ```
    export DATASET_DIR=<directory where the dataset will be saved>
    ./download_dataset.sh
    ```

    #### Set the environment variables
    ```
    export OUTPUT_DIR=<path to the directory where log files will be written>
    export PRECISION=<select one precision: fp32, int8, bf32, bf16, or fp16>

    # Optional environemnt variables:
    export BATCH_SIZE=<set a value for batch size, else it will run with default batch size>
    ```

    #### Run quickstart script
    `accuracy.sh`, `inference_realtime.sh` and `inference_throughput.sh` runs with coco dataset.
    ```
    ./<script.sh>
    ```
    **NOTE**: [`yolov7_int8_default_qparams.json`](/models/object_detection/pytorch/yolov7/yolov7_int8_default_qparams.json) is a default qparams json file for int8, which including the quantization state, such as scales, zero points and inference dtype.
    If you want to tuning accuracy for IPEX int8, you can do the int8 calibration:
    ```
    ./calibration.sh <file where to save the calibrated file> <steps to run calibration>
    ```
    For eaxmple: `./calibration.sh new_int8_config.json 10`

<!--- 80. License -->
## License
[LICENSE](https://github.com/IntelAI/models/blob/master/LICENSE)
