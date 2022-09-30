# Wide ResNet-101-2 Inference

## Description

This document has instructions for running Wide ResNet-101-2 inference.

## Datasets

### ImageNet

The [ImageNet](http://www.image-net.org/) validation dataset is used to run Wide ResNet-101-2
accuracy tests.

Download and extract the ImageNet2012 dataset from [http://www.image-net.org/](http://www.image-net.org/),
then move validation images to labeled subfolders, using
[the valprep.sh shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

After running the data prep script, your folder structure should look something like this:
```
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

* Set Jemalloc Preload for better performance

  After [Jemalloc setup](/docs/general/pytorch/BareMetalSetup.md#build-jemalloc), set the following environment variables.
  ```
  export LD_PRELOAD="path/lib/libjemalloc.so":$LD_PRELOAD
  export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
  ```

* Set IOMP preload for better performance

  IOMP should be installed in your conda env. Set the following environment variables.
  ```
  export LD_PRELOAD=path/lib/libiomp5.so:$LD_PRELOAD
  ```

* Set ENV to use AMX if you are using SPR
  ```
  export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
  ```

* Run the model:
  ```
  cd models

  # Set environment variables
  export DATASET_DIR=<path_to_Imagenet_Dataset>
  export OUTPUT_DIR=<path to the directory where log files will be written>

  # Run a quickstart script (for example, FP32 batch inference)
  bash quickstart/image_recognition/pytorch/wide_resnet101_2/inference/cpu/batch_inference_baremetal.sh fp32
  ```

## Run on Windows
If not already setup, please follow instructions for [environment setup on Windows](/docs/general/Windows.md).

Using Windows CMD.exe, run:
```
cd models

# Env vars
set DATASET_DIR=<path to the Imagenet Dataset>
set OUTPUT_DIR=<path to the directory where log files will be written>

#Run a quickstart script for fp32 precision(FP32 online inference or batch inference or accuracy)
bash quickstart\image_recognition\pytorch\wide_resnet101_2\inference\cpu\batch_inference_baremetal.sh fp32
```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)
