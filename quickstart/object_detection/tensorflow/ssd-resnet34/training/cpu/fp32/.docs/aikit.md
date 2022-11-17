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

Running <model name> <mode> uses code from the
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
export OUTPUT_DIR=<path to the directory where log and checkpoint files will be written>
export MPI_NUM_PROCESSES=<number of MPI processes (optional, defaults to 1)>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

./quickstart/object_detection/tensorflow/ssd-resnet34/training/cpu/fp32/fp32_training.sh
```
