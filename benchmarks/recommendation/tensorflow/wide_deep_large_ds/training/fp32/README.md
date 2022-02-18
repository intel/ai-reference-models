<!--- 0. Title -->
# Wide and Deep Large Dataset FP32 training

<!-- 10. Description -->

This document has instructions for training [Wide and Deep](https://arxiv.org/pdf/1606.07792.pdf)
using a large dataset using Intel-optimized TensorFlow.


<!--- 30. Datasets -->
## Dataset

The Large [Kaggle Display Advertising Challenge Dataset](https://www.kaggle.com/c/criteo-display-ad-challenge/data)
will be used for training Wide and Deep. The [data](https://www.kaggle.com/c/criteo-display-ad-challenge/data) is from
[Criteo](https://www.criteo.com) and has a field indicating if an ad was
clicked (1) or not (0), along with integer and categorical features.

Download the Large Kaggle Display Advertising Challenge Dataset from [Criteo Labs](http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/) in `$DATASET_DIR`.
If the evaluation/train dataset were not available in the above link, it can be downloaded as follow:
   ```
    export DATASET_DIR=<location where dataset files will be saved>
    mkdir $DATASET_DIR && cd $DATASET_DIR
    wget https://storage.googleapis.com/dataset-uploader/criteo-kaggle/large_version/eval.csv
    wget https://storage.googleapis.com/dataset-uploader/criteo-kaggle/large_version/train.csv
   ```
The `DATASET_DIR` environment variable will be used as the dataset directory when running [quickstart scripts](#quick-start-scripts).

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`fp32_training_check_accuracy.sh`](fp32_training_check_accuracy.sh) | Trains the model for a specified number of steps (default is 500) and then compare the accuracy against the accuracy specified in the `TARGET_ACCURACY` env var (ex: `export TARGET_ACCURACY=0.75`). If the accuracy is not met, then script exits with error code 1. The `CHECKPOINT_DIR` environment variable can optionally be defined to start training based on previous set of checkpoints. |
| [`fp32_training.sh`](fp32_training.sh) | Trains the model for 10 epochs. The `CHECKPOINT_DIR` environment variable can optionally be defined to start training based on previous set of checkpoints. |
| [`fp32_training_demo.sh`](fp32_training_demo.sh) | A short demo run that trains the model for 100 steps. |

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
        <li>numactl
        <li>wget
        <li>Activate the `tensorflow` conda environment
        <pre>conda activate tensorflow</pre>
      </ul>
    </td>
    <td>
      <p>To run without AI Kit you will need:</p>
      <ul>
        <li>Python 3
        <li>intel-tensorflow>=2.5.0
        <li>numactl
        <li>git
        <li>wget
        <li>A clone of the Model Zoo repo<br />
        <pre>git clone https://github.com/IntelAI/models.git</pre>
      </ul>
    </td>
  </tr>
</table>

After the setup is complete, set environment variables for the path to your
dataset directory and an output directory where logs will be written. You can
optionally provide a directory where checkpoint files will be read and
written. Navigate to your model zoo directory, then select a
[quickstart script](#quick-start-scripts) to run. Note that some quickstart
scripts might use other environment variables in addition to the ones below,
like `STEPS` and `TARGET_ACCURACY` for the `fp32_training_check_accuracy.sh` script.
```
# cd to your model zoo directory
cd models

export DATASET_DIR=<path to the dataset directory>
export OUTPUT_DIR=<directory where the logs and the saved model will be written>
export CHECKPOINT_DIR=<Optional directory where checkpoint files will be read and written>

./quickstart/recommendation/tensorflow/wide_deep_large_ds/training/cpu/fp32/<script name>.sh
```

<!--- 90. Resource Links-->
## Additional Resources

* To run more advanced use cases, see the instructions [here](Advanced.md)
  for calling the `launch_benchmark.py` script directly.
* To run the model using docker, please see the [oneContainer](http://software.intel.com/containers)
  workload container:<br />
  [https://software.intel.com/content/www/us/en/develop/articles/containers/wide-deep-large-dataset-fp32-training-tensorflow-container.html](https://software.intel.com/content/www/us/en/develop/articles/containers/wide-deep-large-dataset-fp32-training-tensorflow-container.html).

