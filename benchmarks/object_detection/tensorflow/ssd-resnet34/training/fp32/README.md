<!--- 0. Title -->
# SSD-ResNet34 FP32 training

<!-- 10. Description -->
## Description

This document has instructions for running SSD-ResNet34 FP32 training using
Intel-optimized TensorFlow.

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
        <li>contextlib2
        <li>cpio
        <li>Cython
        <li>horovod
        <li>jupyter
        <li>lxml
        <li>matplotlib
        <li>numpy>=1.17.4
        <li>opencv
        <li>openmpi
        <li>openssh
        <li>pillow>=8.1.2
        <li>protoc
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
        <li><a href="https://pypi.org/project/intel-tensorflow/">intel-tensorflow>=2.5.0</a>
        <li>contextlib2
        <li>cpio
        <li>Cython
        <li>horovod
        <li>jupyter
        <li>lxml
        <li>matplotlib
        <li>numpy>=1.17.4
        <li>opencv
        <li>openmpi
        <li>openssh
        <li>pillow>=8.1.2
        <li>protoc
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

Running SSD-ResNet34 training uses code from the
[TensorFlow Model Garden](https://github.com/tensorflow/models).
Clone the  repo at the commit specified below, and set the `TF_MODELS_DIR` environment
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

Set the `DATASET_DIR` to point to the directory with COCO training TF records
files and the `OUTPUT_DIR` to the location where log and checkpoint files will
be written. Use an empty output directory to prevent checkpoint file conflicts
from previous runs. You can optionally set the `MPI_NUM_PROCESSES` environment
variable (defaults to 1). After all the setup is complete, run the
[quickstart script](#quick-start-scripts).
```
# cd to your model zoo directory
cd models

export TF_MODELS_DIR=<path to your clone of the TensorFlow models repo>
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log and checkpoint files will be written>
export MPI_NUM_PROCESSES=<number of MPI processes (optional, defaults to 1)>

./quickstart/object_detection/tensorflow/ssd-resnet34/training/cpu/fp32/fp32_training.sh
```

<!--- 90. Resource Links-->
## Additional Resources

* To run more advanced use cases, see the instructions [here](Advanced.md)
  for calling the `launch_benchmark.py` script directly.
* To run the model using docker, please see the [oneContainer](http://software.intel.com/containers)
  workload container:<br />
  [https://software.intel.com/content/www/us/en/develop/articles/containers/ssd-resnet34-fp32-training-tensorflow-container.html](https://software.intel.com/content/www/us/en/develop/articles/containers/ssd-resnet34-fp32-training-tensorflow-container.html).

