<!--- 50. AI Kit -->
## Run the model
Install the following pre-requisites:
* Python version 3.9
* Create and activate virtual environment.
  ```bash
  virtualenv -p python <virtualenv_name>
  source <virtualenv_name>/bin/activate
  ```
* Install TensorFlow and Intel® Extension for TensorFlow (ITEX):

  Intel® Extension for TensorFlow requires stock TensorFlow v2.10.0 to be installed 
  
  ```bash
  pip install tensorflow==2.10.0
  pip install --upgrade intel-extension-for-tensorflow[gpu]
  ```
   To verify that TensorFlow and ITEX are correctly installed:
  ```
  python -c "import intel_extension_for_tensorflow as itex; print(itex.__version__)"
  ```
* Download the frozen graph model file, and set the FROZEN_GRAPH environment variable to point to where it was saved:
  ```bash
  wget https://storage.googleapis.com/intel-optimized-tensorflow/models/gpu/ssd_mobilenet_v1_int8_itex.pb
  ```
* Install model specific dependencies:
  ```bash
  pip install pycocotools
  ```

See the [datasets section](#datasets) of this document for instructions on
downloading and preprocessing the ImageNet dataset. The path to the ImageNet
TF records files will need to be set as the `DATASET_DIR` environment variable
prior to running a [quickstart script](#quick-start-scripts).

### To run the model on Baremetal
This snippet shows how to run a quickstart script:
```
export DATASET_DIR=<path to the preprocessed COCO TF dataset>
export OUTPUT_DIR=<path to where output log files will be written>
export PRECISION=int8
export FROZEN_GRAPH=<path to pretrained model file (*.pb)>

Run quickstart script:
./quickstart/object_detection/tensorflow/ssd-mobilenet/inference/gpu/<script name>.sh
```