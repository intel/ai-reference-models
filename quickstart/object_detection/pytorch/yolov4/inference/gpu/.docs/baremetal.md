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
* Navigate to Yolov4 inference directory and install model specific dependencies for the workload:
  ```bash
  cd quickstart/object_detection/pytorch/yolov4/inference/gpu
  ./setup.sh
  cd -
  ```
* Download the pretrained weights file, and set the PRETRAINED_MODEL environment variable to point to where it was saved:
  ```bash
  wget https://drive.google.com/open?id=1wv_LiFeCRYwtpkqREPeI13-gPELBDwuJ
  ```

### Run the model on Baremetal
```
Set environment variables:
export DATASET_DIR=<path where yolov4 COCO dataset>
export PRETRAINED_MODEL=<path to directory where the pretrained weights file was saved>
export OUTPUT_DIR=<Path to save the output logs>

Run the inference script, only int8 precision is supported:
./quickstart/object_detection/pytorch/yolov4/inference/gpu/inference_with_dummy_data.sh
```
