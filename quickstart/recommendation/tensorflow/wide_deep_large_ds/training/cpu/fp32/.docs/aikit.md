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
export OUTPUT_DIR=<path to the directory where the logs and the saved model will be written>
export CHECKPOINT_DIR=<Optional directory where checkpoint files will be read and written>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

./quickstart/recommendation/tensorflow/wide_deep_large_ds/training/cpu/fp32/<script name>.sh
```
