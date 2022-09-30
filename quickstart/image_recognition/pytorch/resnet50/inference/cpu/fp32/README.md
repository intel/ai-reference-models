<!--- 0. Title -->
# ResNet50 FP32 inference

<!-- 10. Description -->
## Description

This document has instructions for running ResNet50 FP32 inference.

<!--- 30. Datasets -->
## Datasets

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

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`fp32_online_inference.sh`](fp32_online_inference.sh) | Runs online inference using synthetic data (batch_size=1). |
| [`fp32_batch_inference.sh`](fp32_batch_inference.sh) | Runs batch inference using synthetic data (batch_size=128). |
| [`fp32_accuracy.sh`](fp32_accuracy.sh) | Measures the model accuracy (batch_size=128). |

<!--- 50. Bare Metal -->
## Bare Metal

Follow the instructions to setup your bare metal environment on Linux or Windows systems, download and
preprocess the dataset, and do the model specific setup. Once all the setup is done,
the Model Zoo can be used to run a [quickstart script](#quick-start-scripts).
Ensure that you have enviornment variables set to point to the dataset directory
and an output directory.

Follow the instructions below to clone the Model Zoo repo before running the model on Linux or Windows:
```
# Clone the Model Zoo repo 
git clone https://github.com/IntelAI/models.git
cd models
```

### Run on Linux
Install the following prerequisites in your environment:
* Python 3
* [intel-extension-for-pytorch](https://github.com/intel/intel-extension-for-pytorch)
* [torchvision==v0.6.1](https://github.com/pytorch/vision/tree/v0.6.1)
* numactl
* Run the model:
    ```
    # Env vars
    export DATASET_DIR=<path to the Imagenet Dataset>
    export OUTPUT_DIR=<path to the directory where log files will be written>

    # Run a quickstart script (for example, FP32 batch inference)
    bash quickstart/image_recognition/pytorch/resnet50/inference/cpu/fp32/fp32_batch_inference.sh 
    ```

### Run on Windows
If not already setup, please follow instructions for [environment setup on Windows](/docs/general/Windows.md).

Using Windows CMD.exe, run:
```
# Env vars
set DATASET_DIR=<path to the Imagenet Dataset>
set OUTPUT_DIR=<path to the directory where log files will be written>

#Run a quickstart script for fp32 precision(for example, FP32 batch inference)
bash quickstart\image_recognition\pytorch\resnet50\inference\cpu\fp32\fp32_batch_inference.sh 
```

<!--- 80. License -->
## License

[LICENSE](/LICENSE)
