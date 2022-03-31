<!--- 0. Title -->
# SSD-ResNet34 FP32 training

<!-- 10. Description -->
## Description

This document has instructions for running SSD-ResNet34 FP32 training using
Intel-optimized TensorFlow.

<!--- 20. Download link -->
## Download link

[ssd-resnet34-fp32-training.tar.gz](https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/ssd-resnet34-fp32-training.tar.gz)

<!--- 30. Datasets -->
## Datasets

SSD-ResNet34 training uses the COCO dataset. Use the following instructions
to download and preprocess the dataset.

1.  Download and extract the 2017 training images and annotations for the
    [COCO dataset](http://cocodataset.org/#home):
    ```bash
    export MODEL_WORK_DIR=$(pwd)

    # Download and extract train images
    wget http://images.cocodataset.org/zips/train2017.zip
    unzip train2017.zip

    # Download and extract annotations
    wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip annotations_trainval2017.zip
    ```

2.  Since we are only using the train and validation dataset in this example,
    we will create an empty directory and empty annotations json file to pass
    as the test directories in the next step.
    ```bash
    # Create an empty dir to pass for validation and test images
    mkdir empty_dir

    # Add an empty .json file to bypass validation/test image preprocessing
    cd annotations
    echo "{ \"images\": {}, \"categories\": {}}" > empty.json
    cd ..
    ```

3. Use the [TensorFlow models repo scripts](https://github.com/tensorflow/models)
   to convert the raw images and annotations to the TF records format.
   ```
   git clone https://github.com/tensorflow/models.git tf_models
   cd tf_models
   git checkout 7a9934df2afdf95be9405b4e9f1f2480d748dc40
   cd ..
   ```

4. Install the prerequisites mentioned in the
   [TensorFlow models object detection installation doc](https://github.com/tensorflow/models/blob/v2.3.0/research/object_detection/g3doc/installation.md#dependencies)
   and run [protobuf compilation](https://github.com/tensorflow/models/blob/v2.3.0/research/object_detection/g3doc/installation.md#protobuf-compilation)
   on the code that was cloned in the previous step.

5. After your envionment is setup, run the conversion script:
   ```
   cd tf_models/research/object_detection/dataset_tools/

   # call script to do conversion
   python create_coco_tf_record.py --logtostderr \
         --train_image_dir="$MODEL_WORK_DIR/train2017" \
         --val_image_dir="$MODEL_WORK_DIR/empty_dir" \
         --test_image_dir="$MODEL_WORK_DIR/empty_dir" \
         --train_annotations_file="$MODEL_WORK_DIR/annotations/instances_train2017.json" \
         --val_annotations_file="$MODEL_WORK_DIR/annotations/empty.json" \
         --testdev_annotations_file="$MODEL_WORK_DIR/annotations/empty.json" \
         --output_dir="$MODEL_WORK_DIR/output"
    ```

    The `coco_train.record-*-of-*` files are what we will use in this training example.
    Set the output of the preprocessing script (`export DATASET_DIR=$MODEL_WORK_DIR/output`)
    when running quickstart scripts.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`fp32_training.sh`](/quickstart/object_detection/tensorflow/ssd-resnet34/training/cpu/fp32/fp32_training.sh) | Runs 100 training steps using mpirun for the specified number of processes (defaults to MPI_NUM_PROCESSES=1).  |

<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3
* contextlib2
* cpio
* Cython
* horovod
* [intel-tensorflow>=2.5.0](https://pypi.org/project/intel-tensorflow/)
* jupyter
* lxml
* matplotlib
* numpy>=1.17.4
* opencv
* openmpi
* openssh
* pillow>=8.1.2
* protoc
* pycocotools
* tensorflow-addons==0.11.0

For more information, see the
[installation instructions](https://github.com/tensorflow/models/blob/8110bb64ca63c48d0caee9d565e5b4274db2220a/research/object_detection/g3doc/installation.md#installation)
for object detection models at the
[TensorFlow Model Garden](https://github.com/tensorflow/models) repository.

After installing the prerequisites, download and untar the model package.
Clone the [TensorFlow Model Garden](https://github.com/tensorflow/models)
repo at the commit specified below, and set the `TF_MODELS_DIR` environment
variable to point to that directory. Set the `DATASET_DIR` to point to the
directory with COCO training TF records files and the `OUTPUT_DIR` to the
location where log and checkpoint files will be written. Use an empty
output directory to prevent checkpoint file conflicts from prevouis runs.
You can optionally set the `MPI_NUM_PROCESSES` environment variable (defaults to 1).
After all the setup is complete, run the [quickstart script](#quick-start-scripts).

```
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log and checkpoint files will be written>
export MPI_NUM_PROCESSES=<number of MPI processes (optional, defaults to 1)>

# Clone the tensorflow/models repo at the specified commit.
# Please note that required commit for this section is different from the one used for dataset preparation.
git clone https://github.com/tensorflow/models.git tf_models
cd tf_models
export TF_MODELS_DIR=$(pwd)
git checkout 8110bb64ca63c48d0caee9d565e5b4274db2220a
cd ..

# Download and extract the model package, then run a quickstart script
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/ssd-resnet34-fp32-training.tar.gz
tar -xzf ssd-resnet34-fp32-training.tar.gz
cd ssd-resnet34-fp32-training

./quickstart/fp32_training.sh
```

<!--- 60. Docker -->
## Docker

The model container includes the scripts and libraries needed to run 
SSD-ResNet34 FP32 training. To run one of the quickstart scripts 
using this container, you'll need to provide volume mounts for the dataset 
and an output directory where the log file will be written. Use an empty
output directory to prevent checkpoint file conflicts from previous runs.
To run more than one process, set the `MPI_NUM_PROCESSES` environment
variable in the container.

```
DATASET_DIR=<path to the dataset>
OUTPUT_DIR=<directory where log and checkpoint files will be written>
MPI_NUM_PROCESSES=<number of MPI processes (optional, defaults to 1)>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env MPI_NUM_PROCESSES=${MPI_NUM_PROCESSES} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -it \
  intel/object-detection:tf-latest-ssd-resnet34-fp32-training \
  /bin/bash quickstart/fp32_training.sh
```

If you are new to docker and are running into issues with the container,
see [this document](https://github.com/IntelAI/models/tree/master/docs/general/docker.md)
for troubleshooting tips.

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

