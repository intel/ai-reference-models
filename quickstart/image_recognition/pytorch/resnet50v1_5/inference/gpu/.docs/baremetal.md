<!--- 50. Baremetal -->
## Run the model
Install the following pre-requisites:
* Python version 3.9
* Create and activate virtual environment.
  ```bash
  virtualenv -p python <virtualenv_name>
  source <virtualenv_name>/bin/activate
  ```
* Install PyTorch and Intel® Extension for PyTorch for GPU (IPEX):
  ```bash
  python -m pip install torch==1.10.0a0 -f https://developer.intel.com/ipex-whl-stable-xpu
  python -m pip install numpy==1.23.4
  python -m pip install intel_extension_for_pytorch==1.10.200+gpu -f https://developer.intel.com/ipex-whl-stable-xpu
  ```
  To verify that PyTorch and IPEX are correctly installed:
  ```bash
  python -c "import torch;print(torch.device('xpu'))"  # Sample output: "xpu"
  python -c "import intel_extension_for_pytorch as ipex;print(ipex.xpu.is_available())"  #Sample output True
  python -c "import intel_extension_for_pytorch as ipex;print(ipex.xpu.has_onemkl())"  # Sample output: True
  ```
* Navigate to ResNet50v1.5 inference directory and install model specific dependencies for the workload:
  ```bash
  cd quickstart/image_recognition/pytorch/resnet50v1_5/inference/gpu
  ./setup.sh
  cd -
  ```

See the [datasets section](#datasets) of this document for instructions on
downloading and preprocessing the ImageNet dataset. The path to the ImageNet
dataset files will need to be set as the `DATASET_DIR` environment variable
prior to running a [quickstart script](#quick-start-scripts).

### Run the model on Baremetal
Set environment variables for the path to your dataset, an output directory, and specify the precision to run the quickstart script:
```
To run with ImageNet data, the dataset directory will need to be specified in addition to an output directory and precision.
export DATASET_DIR=<path to the preprocessed imagenet dataset>
export OUTPUT_DIR=<Path to save the output logs>
export PRECISION=int8

# Run a quickstart script
./quickstart/image_recognition/pytorch/resnet50v1_5/inference/gpu/inference_block_format.sh
```
