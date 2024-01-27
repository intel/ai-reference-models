# PyTorch ResNet50 inference

## Description

This document has instructions for running ResNet50 inference using
Intel-optimized PyTorch.

## Dataset

The [ImageNet](http://www.image-net.org/) validation dataset is used when
testing accuracy. The inference scripts use synthetic data, so no dataset
is needed.

Download and extract the ImageNet2012 dataset from http://www.image-net.org/,
then move validation images to labeled subfolders, using
[the valprep.sh shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

The accuracy script looks for a folder named `val`, so after running the
data prep script, your folder structure should look something like this:

```
imagenet
└── val
    ├── ILSVRC2012_img_val.tar
    ├── n01440764
    │   ├── ILSVRC2012_val_00000293.JPEG
    │   ├── ILSVRC2012_val_00002138.JPEG
    │   ├── ILSVRC2012_val_00003014.JPEG
    │   ├── ILSVRC2012_val_00006697.JPEG
    │   └── ...
    └── ...
```
The folder that contains the `val` directory should be set as the
`DATASET_DIR` when running accuracy
(for example: `export DATASET_DIR=/home/<user>/imagenet`).

## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`inference_realtime.sh`](/quickstart/image_recognition/pytorch/resnet50/inference/cpu/inference_realtime.sh) | Runs inference using synthetic data (batch_size=1) for the specified precision (fp32, avx-fp32, int8, avx-int8, bf16, or bf32). |
| [`inference_throughput.sh`](/quickstart/image_recognition/pytorch/resnet50/inference/cpu/inference_throughput.sh) | Runs inference to get the throughput using synthetic data for the specified precision (fp32, avx-fp32, int8, avx-int8, bf16, or bf32). |
| [`accuracy.sh`](/quickstart/image_recognition/pytorch/resnet50/inference/cpu/accuracy.sh) | Measures the model accuracy (batch_size=128) for the specified precision (fp32, avx-fp32, int8, avx-int8, bf16, or bf32). |

> Note: The `avx-int8` and `avx-fp32` precisions run the same scripts as `int8` and `fp32`, except that the
> `DNNL_MAX_CPU_ISA` environment variable is unset. The environment variable is
> otherwise set to `DNNL_MAX_CPU_ISA=AVX512_CORE_AMX`.
* Set ENV to use AMX:
  ```
  export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
  ```

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
  ```

* Run the model:
    
    #### Set the environment variables
    ```bash
    export OUTPUT_DIR=<path to the directory where log files will be written>
    export PRECISION=<select one precision: fp32, avx-fp32, int8, bf32, avx-int8, or bf16>

    # Optional environemnt variables:
    export BATCH_SIZE=<set a value for batch size, else it will run with default batch size>

    # [optional] Compile model with PyTorch Inductor backend
    export TORCH_INDUCTOR=1
    ```

    #### Run quickstart script:
    ```
    NOTE: `inference_realtime.sh` and `inference_throughput.sh` runs with synthetic data
    ./quickstart/image_recognition/pytorch/resnet50/inference/cpu/<script.sh>
    ```

    #### `accuracy.sh` script runs with the Imagenet Dataset:
    ```
    export DATASET_DIR=<path to the Imagenet Dataset>
    ./quickstart/image_recognition/pytorch/resnet50/inference/cpu/accuracy.sh
    ```

## License

[LICENSE](/LICENSE)
