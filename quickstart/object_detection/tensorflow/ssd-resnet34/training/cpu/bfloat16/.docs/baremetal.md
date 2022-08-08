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
wget <package url>
tar -xzf <package name>

# Apply the patch the the TF_MODELS_DIR
cd ${TF_MODELS_DIR}
git apply ../<package dir>/models/object_detection/tensorflow/ssd-resnet34/training/bfloat16/tf-2.0.diff
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
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

cd <package dir>
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
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

cd <package dir>
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
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

cd <package dir>
./quickstart/bfloat16_training_accuracy.sh
```
