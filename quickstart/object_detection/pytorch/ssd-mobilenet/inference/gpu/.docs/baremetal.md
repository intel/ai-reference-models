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
* Navigate to SSD-Mobilenet inference directory and install model specific dependencies for the workload:
  ```bash
  cd quickstart/object_detection/pytorch/ssd-mobilenet/inference/gpu
  ./setup.sh
  cd -
  ```
* Download the dataset label file, and set the "label" environment variable to point to where it was saved (for example: `export label=/home/<user>/voc-model-labels.txt`):
  ```bash
  wget https://storage.googleapis.com/models-hao/voc-model-labels.txt
  ```

This snippet shows how to run the inference quickstart script. The inference script
will download the model weights to the directory location set in 'PRETRAINED_MODEL'.

```
### Run the model on Baremetal
Set environment variables:
export DATASET_DIR=<Path to the VOC2007 folder>
export OUTPUT_DIR=<Path to save the output logs>
export PRETRAINED_MODEL=<path to directory where the model weights will be loaded>
export label=<path to label.txt file>

Run the inference script, only int8 precision is supported:
./quickstart/object_detection/pytorch/ssd-mobilenet/inference/gpu/inference_with_dummy_data.sh
```
