<!--- 0. Title -->
# PyTorch ResNext101 32x16d inference

<!-- 10. Description -->
## Description

This document has instructions for running ResNext101 32x16d inference.

## Datasets

### ImageNet

The [ImageNet](http://www.image-net.org/) validation dataset is used to run ResNext101 32x16d
accuracy tests.

Download and extract the ImageNet2012 dataset from [http://www.image-net.org/](http://www.image-net.org/),
then move validation images to labeled subfolders, using
[the valprep.sh shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

After running the data prep script, your folder structure should look something like this:

```txt
imagenet
└── val
    ├── ILSVRC2012_img_val.tar
    ├── n01440764
    │   ├── ILSVRC2012_val_00000293.JPEG
    │   ├── ILSVRC2012_val_00002138.JPEG
    │   ├── ILSVRC2012_val_00003014.JPEG
    │   ├── ILSVRC2012_val_00006697.JPEG
    │   └── ...
    └── ...
```

The folder that contains the `val` directory should be set as the
`DATASET_DIR` (for example: `export DATASET_DIR=/home/<user>/imagenet`).

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `inference_realtime.sh` | Runs multi instance realtime inference using 4 cores per instance with synthetic data for the specified precision (fp32, avx-fp32, int8, avx-int8, bf16, or bf32). |
| `inference_throughput.sh` | Runs multi instance batch inference using 1 instance per socket with synthetic data for the specified precision (fp32, avx-fp32m int8, avx-int8, bf16, or bf32). |
| `accuracy.sh` | Measures the inference accuracy (providing a `DATASET_DIR` environment variable is required) for the specified precision (fp32, avx-fp32, int8, avx-int8, bf16, or bf32). |

> Note: The `avx-int8` and `avx-fp32` precisions run the same scripts as `int8` and `fp32`, except that the
> `DNNL_MAX_CPU_ISA` environment variable is unset. The environment variable is
> otherwise set to `DNNL_MAX_CPU_ISA=AVX512_CORE_AMX`.
* Set ENV to use AMX:
  ```
  export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
  ```

## Run on Linux

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
  Ensure that you have a clone of the [Intel® AI Reference Models Github repository](https://github.com/IntelAI/models).
  ```
  git clone https://github.com/IntelAI/models.git
  ```

* Run the model:
  ```
  cd models

  # Set environment variables
  export OUTPUT_DIR=<path to the directory where log files will be written>
  export PRECISION=<select precision to run: fp32, avx-fp32, int8, avx-int8, bf32 or bf16>

  # Optional environment variable:
  export BATCH_SIZE=<set a value for batch size else it will run with the default value>

  # Run a quickstart script 
  ./quickstart/image_recognition/pytorch/resnext-32x16d/inference/cpu/<quickstart_script.sh>

  # Set `DATASET_DIR` only for running the `accuracy.sh` script:
  set DATASET_DIR=<path to the Imagenet Dataset>

  ./quickstart/image_recognition/pytorch/resnext-32x16d/inference/cpu/accuracy.sh
  ```

## Run on Windows
If not already setup, please follow instructions for [environment setup on Windows](/docs/general/Windows.md).

Using Windows CMD.exe, run:
```
cd models

# Set environment variables
export OUTPUT_DIR=<path to the directory where log files will be written>
export PRECISION=<select precision to run: fp32, avx-fp32, int8, avx-int8, bf32 or bf16>

# Optional environment variable:
export BATCH_SIZE=<set a value for batch size else it will run with the default value>

# Run a quickstart script 
bash quickstart\image_recognition\pytorch\resnext-32x16d\inference\cpu\<quickstart_script.sh>

# Set `DATASET_DIR` only for running the `accuracy.sh` script:
set DATASET_DIR=<path to the Imagenet Dataset>

bash quickstart\image_recognition\pytorch\resnext-32x16d\inference\cpu\accuracy.sh
```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)
