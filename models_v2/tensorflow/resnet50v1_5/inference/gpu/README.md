# ResNet50v1.5 Model Inference For ITEX 
Best known method of ResNet50v1.5 Inference(float32, float16, bfloat16, tensorfloat32, int8) for intel-extension-for-tensorflow.

## Model Information
| **Use Case** |**Framework** | **Model Repo** | **Branch/Commit/Tag** | **PB**
| :---: | :---: | :---: | :---: | :---: |
| inference | Tensorflow |[IntelAI/Models](https://github.com/IntelAI/models) | master | [resnet50_v1.pb](https://storage.googleapis.com/intel-optimized-tensorflow/models/3_1/resnet50_v1.pb) [resnet50_v1_int8.pb](https://storage.googleapis.com/intel-optimized-tensorflow/models/3_1/resnet50_v1_int8.pb) | |

**Note:** Model files were copied and modified in this directory.
<br>

# Pre-Requisite
* Host has Intel® Data Center GPU MAX or FLEX
* Host has installed latest Intel® Data Center GPU Max & Flex Series Drivers https://dgpu-docs.intel.com/driver/installation.html

# Dataset 
  to download and preprocess the ImageNet validation
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
1. Run `setup.sh` this will install all the required dependencies & create virtual environment `venv`.
2. Activate virtual env: `. ./venv/bin/activate`
3. Setup required environment paramaters
#### Set Model Parameters
Export those parameters to script or environment.
| **Parameter** | **export |
| :---: | :--- |
| **DATASET_DIR** | `export DATASET_PATH=/the/path/to/ImageNet` (accuracy mode need) |
| **PB_FILE_PATH** | `export PB_FILE_PATH=/the/path/to/xx.pb` (int8:resnet50_v1_int8.pb, others:resnet50_v1.pb)|
| **BATCH_SIZE** | `export BATCH_SIZE=1024` (optional, default is 1024) |
| **TEST_TYPE** | `export TEST_MODE=inference` (inference or accuracy) |
| **DTYPE** | `export PRECISION=float32` (float32,tensorfloat32,float16 ,bfloat16 or int8) |         |
4. Run `run_model.sh`

## Output

Output will typicall looks like:
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
