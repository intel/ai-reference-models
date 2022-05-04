<!--- 0. Title -->
# SSD-ResNet34 BFloat16 training

<!-- 10. Description -->
## Description

This document has instructions for running SSD-ResNet34 BFloat16 training using
Intel-optimized TensorFlow.

<!--- 20. Download link -->
## Download link

[ssd-resnet34-bfloat16-training.tar.gz](https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/ssd-resnet34-bfloat16-training.tar.gz)

<!--- 30. Datasets -->
## Datasets

SSD-ResNet34 training uses the COCO training dataset. Use the following instructions
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

For accuracy testing, download the COCO validation dataset, using the
[instructions here](/datasets/coco).

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`bfloat16_training_demo.sh`](/quickstart/object_detection/tensorflow/ssd-resnet34/training/cpu/bfloat16/bfloat16_training_demo.sh) | Executes a demo run with a limited number of training steps to test performance. Set the number of steps using the `TRAIN_STEPS` environment variable (defaults to 100). |
| [`bfloat16_training.sh`](/quickstart/object_detection/tensorflow/ssd-resnet34/training/cpu/bfloat16/bfloat16_training.sh) | Runs multi-instance training to convergence. Download the backbone model specified in the instructions below and pass that directory path in the `BACKBONE_MODEL_DIR` environment variable. |
| [`bfloat16_training_accuracy.sh`](/quickstart/object_detection/tensorflow/ssd-resnet34/training/cpu/bfloat16/bfloat16_training_accuracy.sh) | Runs the model in eval mode to check accuracy. Specify which checkpoint files to use with the `CHECKPOINT_DIR` environment variable. |

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
variable to point to that directory.
```
# Clone the tensorflow/models repo at the specified commit.
# Please note that required commit for this section is different from the one used for dataset preparation.
git clone https://github.com/tensorflow/models.git tf_models
cd tf_models
export TF_MODELS_DIR=$(pwd)
git checkout 8110bb64ca63c48d0caee9d565e5b4274db2220a
cd ..
```

Download and untar the model training package and apply a patch to the TensorFlow models code for TF 2.0:
```
# Download and extract the model package, then run a quickstart script
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/ssd-resnet34-bfloat16-training.tar.gz
tar -xzf ssd-resnet34-bfloat16-training.tar.gz

# Apply the patch the the TF_MODELS_DIR
cd ${TF_MODELS_DIR}
git apply ../ssd-resnet34-bfloat16-training/models/object_detection/tensorflow/ssd-resnet34/training/bfloat16/tf-2.0.diff
cd ..
```

To run the [`bfloat16_training_demo.sh`](bfloat16_training_demo.sh) quickstart
script, set the `OUTPUT_DIR` (location where you want log and checkpoint files to be written)
and `DATASET_DIR` (path to the COCO training dataset). Use an empty output
directory to prevent conflict with checkpoint files from previous runs. You can optionally
set the `TRAIN_STEPS` (defaults to 100) and `MPI_NUM_PROCESSES` (defaults to 1).
```
export DATASET_DIR=<path to the COCO training data>
export OUTPUT_DIR=<directory where the log and checkpoint files will be written>
export TRAIN_STEPS=<optional, defaults to 100>
export MPI_NUM_PROCESSES=<optional, defaults to 1>

cd ssd-resnet34-bfloat16-training
./quickstart/bfloat16_training_demo.sh
```

To run training and achieve convergence, download the backbone model using the
commands below and set your download directory path as the `BACKBONE_MODEL_DIR`.
Again, the `DATASET_DIR` should point to the COCO training dataset and the
`OUTPUT_DIR` is the location where log and checkpoint files will be written.
You can optionally set the `MPI_NUM_PROCESSES` (defaults to 4).
```
export BACKBONE_MODEL_DIR="$(pwd)/backbone_model"
mkdir -p $BACKBONE_MODEL_DIR
wget -P $BACKBONE_MODEL_DIR https://storage.googleapis.com/intel-optimized-tensorflow/models/ssd-backbone/checkpoint
wget -P $BACKBONE_MODEL_DIR https://storage.googleapis.com/intel-optimized-tensorflow/models/ssd-backbone/model.ckpt-28152.data-00000-of-00001
wget -P $BACKBONE_MODEL_DIR https://storage.googleapis.com/intel-optimized-tensorflow/models/ssd-backbone/model.ckpt-28152.index
wget -P $BACKBONE_MODEL_DIR https://storage.googleapis.com/intel-optimized-tensorflow/models/ssd-backbone/model.ckpt-28152.meta

export DATASET_DIR=<path to the COCO training data>
export OUTPUT_DIR=<directory where the log file and checkpoints will be written>
export MPI_NUM_PROCESSES=<optional, defaults to 4>

cd ssd-resnet34-bfloat16-training
./quickstart/bfloat16_training.sh
```

To run in eval mode (to check accuracy), set the `CHECKPOINT_DIR` to the
directory where your checkpoint files are located, set the `DATASET_DIR` to
the COCO validation dataset location, and the `OUTPUT_DIR` to the location
where log files will be written. You can optionally set the `MPI_NUM_PROCESSES`
(defaults to 1).
```
export DATASET_DIR=<path to the COCO validation data>
export OUTPUT_DIR=<directory where the log file will be written>
export CHECKPOINT_DIR=<directory where your checkpoint files are located>
export MPI_NUM_PROCESSES=<optional, defaults to 1>

cd ssd-resnet34-bfloat16-training
./quickstart/bfloat16_training_accuracy.sh
```

<!--- 60. Docker -->
## Docker

The model container includes the scripts and libraries needed to run 
SSD-ResNet34 BFloat16 training. To run one of the quickstart scripts 
using this container, you'll need to provide volume mounts for the dataset 
and an output directory where the log files and checkpoints will be written.
Use an empty output directory to prevent conflicts with checkpoint files
from previous runs. To run more than one process, set the `MPI_NUM_PROCESSES` environment
variable in the container. Depending on which quickstart script is being
run, other volume mounts or environment variables may be required.

When using the [`bfloat16_training_demo.sh`](bfloat16_training_demo.sh)
quickstart script, the `TRAIN_STEPS` (defaults to 100) environment variable
can be set in addition to the `DATASET_DIR` and `OUTPUT_DIR`. The
`MPI_NUM_PROCESSES` will default to 1 if it is not set.
```
export DATASET_DIR=<path to the COCO training data>
export OUTPUT_DIR=<directory where the log and checkpoint file will be written>
export TRAIN_STEPS=<optional, defaults to 100>
export MPI_NUM_PROCESSES=<optional, defaults to 1>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env TRAIN_STEPS=${TRAIN_STEPS} \
  --env MPI_NUM_PROCESSES=${MPI_NUM_PROCESSES} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -it \
  intel/object-detection:tf-latest-ssd-resnet34-bfloat16-training \
  /bin/bash quickstart/bfloat16_training_demo.sh
```

To run the [`bfloat16_training.sh`](bfloat16_training.sh) quickstart script,
download the backbone model using the commands below. This directory where
the backbone model files are saved to is the `BACKBONE_MODEL_DIR` which will
get mounted in the container and set as an environment variable, just like
the `DATASET_DIR` and `OUTPUT_DIR`. The `MPI_NUM_PROCESSES` will default
to 4 if it is not set.
```
export BACKBONE_MODEL_DIR="$(pwd)/backbone_model"
mkdir -p $BACKBONE_MODEL_DIR
wget -P $BACKBONE_MODEL_DIR https://storage.googleapis.com/intel-optimized-tensorflow/models/ssd-backbone/checkpoint
wget -P $BACKBONE_MODEL_DIR https://storage.googleapis.com/intel-optimized-tensorflow/models/ssd-backbone/model.ckpt-28152.data-00000-of-00001
wget -P $BACKBONE_MODEL_DIR https://storage.googleapis.com/intel-optimized-tensorflow/models/ssd-backbone/model.ckpt-28152.index
wget -P $BACKBONE_MODEL_DIR https://storage.googleapis.com/intel-optimized-tensorflow/models/ssd-backbone/model.ckpt-28152.meta

export DATASET_DIR=<path to the COCO training data>
export OUTPUT_DIR=<directory where the log file and checkpoints will be written>
export MPI_NUM_PROCESSES=<optional, defaults to 4>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env BACKBONE_MODEL_DIR=${BACKBONE_MODEL_DIR} \
  --env MPI_NUM_PROCESSES=${MPI_NUM_PROCESSES} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --volume ${BACKBONE_MODEL_DIR}:${BACKBONE_MODEL_DIR} \
  --privileged --init -it \
  intel/object-detection:tf-latest-ssd-resnet34-bfloat16-training \
  /bin/bash quickstart/bfloat16_training.sh
```

To run the [`bfloat16_training_accuracy.sh`](bfloat16_training_accuracy.sh)
quickstart script, set the `CHECKPOINT_DIR` to the directory where your
checkpoint files are located. The `CHECKPOINT_DIR` needs to get mounted in
the container and set as an environment variable, just like the `DATASET_DIR`
and `OUTPUT_DIR`. Note that when testing accuracy, the `DATASET_DIR` points
to the COCO validation dataset, instead of the training dataset. The
`MPI_NUM_PROCESSES` will default to 1 if it is not set.
```
export DATASET_DIR=<path to the COCO validation data>
export OUTPUT_DIR=<directory where the log file will be written>
export CHECKPOINT_DIR=<directory where your checkpoint files are located>
export MPI_NUM_PROCESSES=<optional, defaults to 1>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env CHECKPOINT_DIR=${CHECKPOINT_DIR} \
  --env MPI_NUM_PROCESSES=${MPI_NUM_PROCESSES} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --volume ${CHECKPOINT_DIR}:${CHECKPOINT_DIR} \
  --privileged --init -it \
  intel/object-detection:tf-latest-ssd-resnet34-bfloat16-training \
  /bin/bash quickstart/bfloat16_training_accuracy.sh
```

If you are new to docker and are running into issues with the container,
see [this document](https://github.com/IntelAI/models/tree/master/docs/general/docker.md)
for troubleshooting tips.

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

