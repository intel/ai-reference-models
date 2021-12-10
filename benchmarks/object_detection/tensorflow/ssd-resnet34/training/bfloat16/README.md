<!--- 0. Title -->
# SSD-ResNet34 BFloat16 training

<!-- 10. Description -->
## Description

This document has instructions for running SSD-ResNet34 BFloat16 training using
Intel-optimized TensorFlow.

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

<!--- 50. AI Kit -->
## Run the model

Setup your environment using the instructions below, depending on if you are
using [AI Kit](/docs/general/tensorflow/AIKit.md):

<table>
  <tr>
    <th>Setup using AI Kit</th>
    <th>Setup without AI Kit</th>
  </tr>
  <tr>
    <td>
      <p>To run using AI Kit you will need:</p>
      <ul>
        <li>git
        <li>numactl
        <li>wget
        <li>contextlib2
        <li>cpio
        <li>Cython
        <li>horovod
        <li>jupyter
        <li>lxml
        <li>matplotlib
        <li>numpy>=1.17.4
        <li>opencv-python
        <li>openmpi
        <li>openssh
        <li>pillow>=8.1.2
        <li>protobuf-compiler
        <li>pycocotools
        <li>tensorflow-addons==0.11.0
        <li>Activate the tensorflow 2.5.0 conda environment
        <pre>conda activate tensorflow</pre>
      </ul>
    </td>
    <td>
      <p>To run without AI Kit you will need:</p>
      <ul>
        <li>Python 3
        <li>git
        <li>numactl
        <li>wget
        <li><a href="https://pypi.org/project/intel-tensorflow/">intel-tensorflow>=2.5.0</a>
        <li>contextlib2
        <li>cpio
        <li>Cython
        <li>horovod
        <li>jupyter
        <li>lxml
        <li>matplotlib
        <li>numpy>=1.17.4
        <li>opencv-python
        <li>openmpi
        <li>openssh
        <li>pillow>=8.1.2
        <li>protobuf-compiler
        <li>pycocotools
        <li>tensorflow-addons==0.11.0
        <li>A clone of the Model Zoo repo<br />
        <pre>git clone https://github.com/IntelAI/models.git</pre>
      </ul>
    </td>
  </tr>
</table>

For more information on the dependencies, see the
[installation instructions](https://github.com/tensorflow/models/blob/8110bb64ca63c48d0caee9d565e5b4274db2220a/research/object_detection/g3doc/installation.md#installation)
for object detection models at the
[TensorFlow Model Garden](https://github.com/tensorflow/models) repository.

Running SSD-ResNet34 training uses code from the [TensorFlow Model Garden](https://github.com/tensorflow/models).
Clone the repo at the commit specified below, and set the `TF_MODELS_DIR`
environment variable to point to that directory. Apply the TF2 patch from
the model zoo to the TensorFlow models directory.
```
# Clone the tensorflow/models repo at the specified commit.
# Please note that required commit for this section is different from the one used for dataset preparation.
git clone https://github.com/tensorflow/models.git tf_models
cd tf_models
export TF_MODELS_DIR=$(pwd)
git checkout 8110bb64ca63c48d0caee9d565e5b4274db2220a

# Apply the patch from the model zoo directory to the TensorFlow Models repo
git apply <model zoo directory>/models/object_detection/tensorflow/ssd-resnet34/training/bfloat16/tf-2.0.diff

# Protobuf compilation from the TF models research directory
cd research
protoc object_detection/protos/*.proto --python_out=.

cd ../..
```

To run the [`bfloat16_training_demo.sh`](bfloat16_training_demo.sh) quickstart
script, set the `OUTPUT_DIR` (location where you want log and checkpoint files to be written)
and `DATASET_DIR` (path to the COCO training dataset). Use an empty output
directory to prevent conflicts with checkpoint files from previous runs. You can optionally
set the `TRAIN_STEPS` (defaults to 100) and `MPI_NUM_PROCESSES` (defaults to 1).
```
# cd to your model zoo directory
cd models

export TF_MODELS_DIR=<path to the clone of the TensorFlow models repo>
export DATASET_DIR=<path to the COCO training data>
export OUTPUT_DIR=<directory where the log and checkpoint files will be written>
export TRAIN_STEPS=<optional, defaults to 100>
export MPI_NUM_PROCESSES=<optional, defaults to 1>

./quickstart/object_detection/tensorflow/ssd-resnet34/training/cpu/bfloat16/bfloat16_training_demo.sh
```

To run training and achieve convergence, download the backbone model using the
commands below and set your download directory path as the `BACKBONE_MODEL_DIR`.
Again, the `DATASET_DIR` should point to the COCO training dataset and the
`OUTPUT_DIR` is the location where log and checkpoint files will be written.
Use an empty `OUTPUT_DIR` to prevent conflicts with previously generated checkpoint
files. You can optionally set the `MPI_NUM_PROCESSES` (defaults to 4).
```
export BACKBONE_MODEL_DIR="$(pwd)/backbone_model"
mkdir -p $BACKBONE_MODEL_DIR
wget -P $BACKBONE_MODEL_DIR https://storage.googleapis.com/intel-optimized-tensorflow/models/ssd-backbone/checkpoint
wget -P $BACKBONE_MODEL_DIR https://storage.googleapis.com/intel-optimized-tensorflow/models/ssd-backbone/model.ckpt-28152.data-00000-of-00001
wget -P $BACKBONE_MODEL_DIR https://storage.googleapis.com/intel-optimized-tensorflow/models/ssd-backbone/model.ckpt-28152.index
wget -P $BACKBONE_MODEL_DIR https://storage.googleapis.com/intel-optimized-tensorflow/models/ssd-backbone/model.ckpt-28152.meta

# cd to your model zoo directory
cd models

export TF_MODELS_DIR=<path to the clone of the TensorFlow models repo>
export DATASET_DIR=<path to the COCO training data>
export OUTPUT_DIR=<directory where the log file and checkpoints will be written>
export MPI_NUM_PROCESSES=<optional, defaults to 4>

./quickstart/object_detection/tensorflow/ssd-resnet34/training/cpu/bfloat16/bfloat16_training.sh
```

To run in eval mode (to check accuracy), set the `CHECKPOINT_DIR` to the
directory where your checkpoint files are located, set the `DATASET_DIR` to
the COCO validation dataset location, and the `OUTPUT_DIR` to the location
where log files will be written. You can optionally set the `MPI_NUM_PROCESSES`
(defaults to 1).
```
# cd to your model zoo directory
cd models

export TF_MODELS_DIR=<path to the clone of the TensorFlow models repo>
export DATASET_DIR=<path to the COCO validation data>
export OUTPUT_DIR=<directory where the log file will be written>
export CHECKPOINT_DIR=<directory where your checkpoint files are located>
export MPI_NUM_PROCESSES=<optional, defaults to 1>

./quickstart/object_detection/tensorflow/ssd-resnet34/training/cpu/bfloat16/bfloat16_training_accuracy.sh
```

<!--- 90. Resource Links-->
## Additional Resources

* To run more advanced use cases, see the instructions [here](Advanced.md)
  for calling the `launch_benchmark.py` script directly.
* To run the model using docker, please see the [oneContainer](http://software.intel.com/containers)
  workload container:<br />
  [https://software.intel.com/content/www/us/en/develop/articles/containers/ssd-resnet34-bfloat16-training-tensorflow-container.html](https://software.intel.com/content/www/us/en/develop/articles/containers/ssd-resnet34-bfloat16-training-tensorflow-container.html).

