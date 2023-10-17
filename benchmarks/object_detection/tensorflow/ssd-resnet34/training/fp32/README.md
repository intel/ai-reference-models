<!--- 0. Title -->
# SSD-ResNet34 FP32 training

<!-- 10. Description -->
## Description

This document has instructions for running SSD-ResNet34 FP32 training using
Intel-optimized TensorFlow.

<!--- 30. Datasets -->
## Datasets

SSD-ResNet34 training uses the COCO training dataset. Use the [instructions](https://github.com/IntelAI/models/tree/master/datasets/coco/README_train.md) to download and preprocess the dataset.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`fp32_training.sh`](/quickstart/object_detection/tensorflow/ssd-resnet34/training/cpu/fp32/fp32_training.sh) | Runs 100 training steps using mpirun for the specified number of processes (defaults to MPI_NUM_PROCESSES=1).  |

<!--- 50. Bare Metal -->
<!--- 50. AI Tools -->
## Run the model

Setup your environment using the instructions below, depending on if you are
using [AI Tools](/docs/general/tensorflow/AITools.md):

<table>
  <tr>
    <th>Setup using AI Tools</th>
    <th>Setup without AI Tools</th>
  </tr>
  <tr>
    <td>
      <p>To run using AI Tools you will need:</p>
      <ul>
        <li>git
        <li>numactl
        <li>contextlib2
        <li>cpio
        <li>Cython
        <li>horovod>=0.27.0
        <li>jupyter
        <li>lxml
        <li>matplotlib
        <li>numpy>=1.17.4
        <li>opencv
        <li>openmpi
        <li>openssh
        <li>pillow>=9.3.0
        <li>protoc
        <li>pycocotools
        <li>tensorflow-addons==0.18.0
        <li>Activate the tensorflow conda environment
        <pre>conda activate tensorflow</pre>
      </ul>
    </td>
    <td>
      <p>To run without AI Tools you will need:</p>
      <ul>
        <li>Python 3
        <li>git
        <li>numactl
        <li><a href="https://pypi.org/project/intel-tensorflow/">intel-tensorflow>=2.5.0</a>
        <li>contextlib2
        <li>cpio
        <li>Cython
        <li>horovod>=0.27.0
        <li>jupyter
        <li>lxml
        <li>matplotlib
        <li>numpy>=1.17.4
        <li>opencv
        <li>openmpi
        <li>openssh
        <li>pillow>=9.3.0
        <li>protoc
        <li>pycocotools
        <li>tensorflow-addons==0.18.0
        <li>A clone of the AI Reference Models repo<br />
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
# cd to your AI Reference Models directory
cd models

export TF_MODELS_DIR=<path to your clone of the TensorFlow models repo>
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<path to the directory where log and checkpoint files will be written>
export MPI_NUM_PROCESSES=<number of MPI processes (optional, defaults to 1)>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

./quickstart/object_detection/tensorflow/ssd-resnet34/training/cpu/fp32/fp32_training.sh
```

<!--- 90. Resource Links-->
## Additional Resources

* To run more advanced use cases, see the instructions [here](Advanced.md)
  for calling the `launch_benchmark.py` script directly.
* To run the model using docker, please see the [oneContainer](https://www.intel.com/content/www/us/en/developer/tools/software-catalog/containers.html)
  workload container:<br />
  [https://www.intel.com/content/www/us/en/developer/articles/containers/ssd-resnet34-fp32-training-tensorflow-container.html](https://www.intel.com/content/www/us/en/developer/articles/containers/ssd-resnet34-fp32-training-tensorflow-container.html).

