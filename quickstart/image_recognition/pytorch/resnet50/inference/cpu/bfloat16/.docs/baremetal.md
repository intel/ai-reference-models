<!--- 50. Bare Metal -->
## Bare Metal

Follow the instructions to setup your bare metal environment on either Linux or Windows systems, download and
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

```
# Env vars
export DATASET_DIR=<path to the Imagenet Dataset>
export OUTPUT_DIR=<path to the directory where log files will be written>

# Run a quickstart script (for example, bfloat16 batch inference)
bash quickstart/image_recognition/pytorch/resnet50/inference/cpu/bfloat16/fp32_batch_inference.sh 
```

### Run the model on Windows
If not already setup, please follow instructions for [environment setup on Windows](/docs/general/Windows.md).

Using Windows CMD.exe, run:
```
# Env vars
set DATASET_DIR=<path to the Imagenet Dataset>
set OUTPUT_DIR=<path to the directory where log files will be written>

#Run a quickstart script for bfloat16 precision(for example, BFloat16 batch inference)
bash quickstart\image_recognition\pytorch\resnet50\inference\cpu\bfloat16\fp32_batch_inference.sh 
```
