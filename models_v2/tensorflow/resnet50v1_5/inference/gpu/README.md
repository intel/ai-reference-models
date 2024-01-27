# ResNet50v1.5 Model Inference with Intel® Extention for TensorFlow
Best known method of ResNet50v1.5 Inference(float32, float16, bfloat16, tensorfloat32, int8) for Intel® Extention for TensorFlow.

## Model Information
| **Use Case** |**Framework** | **Model Repo** | **Branch/Commit/Tag** | **PB**
| :---: | :---: | :---: | :---: | :---: |
| inference | Tensorflow |[IntelAI/Models](https://github.com/IntelAI/models) | master | [resnet50_v1.pb](https://storage.googleapis.com/intel-optimized-tensorflow/models/3_1/resnet50_v1.pb) [resnet50_v1_int8.pb](https://storage.googleapis.com/intel-optimized-tensorflow/models/3_1/resnet50_v1_int8.pb) | |

**Note:** Model files were copied and modified in this directory. Refer to [CONTAINER_FLEX.md](CONTAINER_FLEX.md) for ResNet50v1.5 Inference instructions using docker containers.

# Pre-Requisite
* Host has Intel® Data Center GPU Flex Series
* Host has installed latest Intel® Data Center GPU Flex Series Drivers https://dgpu-docs.intel.com/driver/installation.html
* The following Intel® oneAPI Base Toolkit components are required:
  - Intel® oneAPI DPC++ Compiler (Placeholder DPCPPROOT as its installation path)
  - Intel® oneAPI Math Kernel Library (oneMKL) (Placeholder MKLROOT as its installation path)
  - Intel® oneAPI MPI Library
  - Intel® oneAPI TBB Library
  - Intel® oneAPI CCL Library

  Follow instructions at [Intel® oneAPI Base Toolkit Download page](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?operatingsystem=linux) to setup the package manager repository.

# Dataset 
To download and preprocess the ImageNet validation
and training images to get them into the TF records format. The scripts used
are based on the conversion scripts from the
[TensorFlow TPU repo](https://github.com/tensorflow/tpu) and have been adapted
to allow for offline preprocessing with cloud storage. Note that downloading
the original images from ImageNet requires registering for an account and
logging in.

1. Go to the [ImageNet webpage (https://image-net.org)](https://image-net.org)
   and log in (or create an account, if you don't have one). After logging in,
   click the link at the top to get to the "Download" page. Select "2012" from
   the list of available datasets.
   Download the following tar files and save it to the compute system
   (e.g. `/home/<user>/imagenet_raw_data`), 500GB disk space is required:

   * Training images (Task 1 & 2). 138GB. MD5: 1d675b47d978889d74fa0da5fadfb00e
   * Validation images (all tasks). 6.3GB. MD5: 29b22e2961454d5413ddabcf34fc5622

   After this step is done, `/home/<user>/imagenet_raw_data` should have the above two
   tar files, one is 138GB (`ILSVRC2012_img_train.tar`) and the other 6.3GB
   (`ILSVRC2012_img_val.tar`).

2. Setup a python 3 virtual environment with TensorFlow and the other
   dependencies specified below. Note that google-cloud-storage is a dependency
   of the script, but these instructions will not be using cloud storage.
   ```
   python3 -m venv tf_env
   source tf_env/bin/activate
   pip install --upgrade pip==19.3.1
   pip install intel-tensorflow
   pip install -I urllib3
   pip install wget
   deactivate
   ```

3. Download and run the [imagenet_to_tfrecords.sh](imagenet_to_tfrecords.sh) script and pass
   arguments for the directory with the ImageNet tar files that were downloaded. To pre-process the entire dataset, pass `training` flag to the following script. To pre-process only the validation dataset, pass `inference` flag to the following script as shown below. 
   in step 1 (e.g. `/home/<user>/imagenet_raw_data`).

   ```bash
   wget https://raw.githubusercontent.com/IntelAI/models/master/datasets/imagenet/imagenet_to_tfrecords.sh
   ./imagenet_to_tfrecords.sh <IMAGENET DIR> inference
   ```
   The `imagenet_to_tfrecords.sh` script extracts the ImageNet tar files, downloads and
   then runs the [`imagenet_to_gcs.py`](imagenet_to_gcs.py) script to convert the
   files to TF records. As the script is running you should see output like:
   ```
   I0911 16:23:59.174904 140581751400256 imagenet_to_gcs.py:354] Finished writing file: <IMAGENET DIR>/tf_records/train/train-00000-of-01024
   I0911 16:23:59.199399 140581751400256 imagenet_to_gcs.py:354] Finished writing file: <IMAGENET DIR>/tf_records/train/train-00001-of-01024
   I0911 16:23:59.221770 140581751400256 imagenet_to_gcs.py:354] Finished writing file: <IMAGENET DIR>/tf_records/train/train-00002-of-01024
   I0911 16:23:59.251754 140581751400256 imagenet_to_gcs.py:354] Finished writing file: <IMAGENET DIR>/tf_records/train/train-00003-of-01024
   ...
   I0911 16:24:22.338566 140581751400256 imagenet_to_gcs.py:402] Processing the validation data.
   I0911 16:24:23.271091 140581751400256 imagenet_to_gcs.py:354] Finished writing file: <IMAGENET DIR>/tf_records/validation/validation-00000-of-00128
   I0911 16:24:24.260855 140581751400256 imagenet_to_gcs.py:354] Finished writing file: <IMAGENET DIR>/tf_records/validation/validation-00001-of-00128
   I0911 16:24:25.179738 140581751400256 imagenet_to_gcs.py:354] Finished writing file: <IMAGENET DIR>/tf_records/validation/validation-00002-of-00128
   I0911 16:24:26.097850 140581751400256 imagenet_to_gcs.py:354] Finished writing file: <IMAGENET DIR>/tf_records/validation/validation-00003-of-00128
   I0911 16:24:27.028785 140581751400256 imagenet_to_gcs.py:354] Finished writing file: <IMAGENET DIR>/tf_records/validation/validation-00004-of-00128
   ...
   ```
   After the `imagenet_to_gcs.py` script completes, the `imagenet_to_tfrecords.sh` script combines
   the train and validation files into the `<IMAGENET DIR>/tf_records`
   directory. The folder should contains 1024 training files and 128 validation
   files.
   ```
   $ ls -1 <IMAGENET DIR>/tf_records/
   train-00000-of-01024
   train-00001-of-01024
   train-00002-of-01024
   train-00003-of-01024
   ...
   validation-00000-of-00128
   validation-00001-of-00128
   validation-00002-of-00128
   validation-00003-of-00128
   ...
   ```
## Inference
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models_v2/tensorflow/resnet50v1_5/inference/gpu`
3. create virtual environment `venv` and activate it:
    ```
    python3 -m venv venv
    . ./venv/bin/activate
    ```
4. Run setup.sh
    ```
    ./setup.sh
    ```
5. Install [tensorflow and ITEX](https://pypi.org/project/intel-extension-for-tensorflow/)
6. Download the PB files (pre trained models) and the set the path for PB_FILE_PATH:
   ```
   # For int8:
   wget https://storage.googleapis.com/intel-optimized-tensorflow/models/3_1/resnet50_v1_int8.pb

   # For float32, tensorflow32, float16 and bfloat16:
   wget https://storage.googleapis.com/intel-optimized-tensorflow/models/3_1/resnet50_v1.pb
   ```
7. Set environment variables for Intel® oneAPI Base Toolkit: 
    Default installation location `{ONEAPI_ROOT}` is `/opt/intel/oneapi` for root account, `${HOME}/intel/oneapi` for other accounts
    ```bash
    source {ONEAPI_ROOT}/compiler/latest/env/vars.sh
    source {ONEAPI_ROOT}/mkl/latest/env/vars.sh
    source {ONEAPI_ROOT}/tbb/latest/env/vars.sh
    source {ONEAPI_ROOT}/mpi/latest/env/vars.sh
    source {ONEAPI_ROOT}/ccl/latest/env/vars.sh
8. Setup required environment paramaters
   #### Set Model Parameters
   Export those parameters to script or environment.
   | **Parameter** | **export** |
   | :---: | :---: |
   | **DATASET_DIR** | `export DATASET_PATH=/the/path/to/ImageNet` (accuracy mode need) |
   | **PB_FILE_PATH** | `export PB_FILE_PATH=/the/path/to/xx.pb` (int8:resnet50_v1_int8.pb, others:resnet50_v1.pb)|
   | **BATCH_SIZE** | `export BATCH_SIZE=1024` (optional, default is 1024) |
   | **TEST_MODE** | `export TEST_MODE=inference` (inference or accuracy)  |
   | **DTYPE** | `export PRECISION=float32` (float32,tensorfloat32,float16 ,bfloat16 or int8) |    
   | **FLEX_GPU_TYPE** |  `export FLEX_GPU_TYPE=<flex_140 or flex_170>`      |
8. Run `run_model.sh`

## Output

Output typically looks like:
#inference
```
Iteration 4997: 0.xxxx sec
Iteration 4998: 0.xxxx sec
Iteration 4999: 0.xxxx sec
Iteration 5000: 0.xxxx sec
Average time: 0.xxxx sec
Batch size = 1024
Throughput: xxxx images/sec
```
#accuracy
```
Iteration time: 0.0042 ms
Processed 49920 images. (Top1 accuracy, Top5 accuracy) = (xxxx, xxxx)
Iteration time: 0.0043 ms
Processed 49952 images. (Top1 accuracy, Top5 accuracy) = (xxxx, xxxx)
Iteration time: 0.0042 ms
Processed 49984 images. (Top1 accuracy, Top5 accuracy) = (xxxx, xxxx)
```

Final results of the training run can be found in `results.yaml` file.
```
results:
 - key: throughput
   value: 23.507529269636105
   unit: images/sec
```
