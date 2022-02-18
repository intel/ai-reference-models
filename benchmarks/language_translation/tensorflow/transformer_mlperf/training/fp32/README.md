<!--- 0. Title -->
# Transformer Language FP32 training

<!-- 10. Description -->
## Description

This document has instructions for running Transformer Language FP32 training in mlperf
Benchmark suits using Intel-optimized TensorFlow.

Detailed information on mlperf Benchmark can be found in [mlperf/training](https://github.com/mlperf/training/tree/master/translation/tensorflow/transformer)

<!--- 30. Datasets -->
## Datasets

Follow [instructions](https://github.com/IntelAI/models/tree/master/datasets/transformer_data/README.md) to download and preprocess the WMT English-German dataset.
Set `DATASET_DIR` to point out to the location of the dataset directory.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

Transformer Language in mlperf benchmark can run with full training or
fewer training steps. During training we can control if it will do the evaluation
or not.

| Script name | Description |
|-------------|-------------|
| [`fp32_training_demo.sh`](/quickstart/language_translation/tensorflow/transformer_mlperf/training/cpu/fp32/fp32_training_demo.sh) | Runs 100 training steps (run on a single socket of the CPU). |
| [`fp32_training.sh`](/quickstart/language_translation/tensorflow/transformer_mlperf/training/cpu/fp32/fp32_training.sh) | Runs 200 training steps, saves checkpoints and do evaluation (run on a single socket of the CPU). |
| [`fp32_training_mpirun.sh`](/quickstart/language_translation/tensorflow/transformer_mlperf/training/cpu/fp32/fp32_training_mpirun.sh) | Runs training in multi-instance mode "2 sockets in a single node for example" using mpirun for the specified number of processes. |

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
        <li>gcc-8 (for multi-instance only)
        <li>g++-8 (for multi-instance only)
        <li>libopenmpi-dev (for multi-instance only)
        <li>openmpi (for multi-instance only)
        <li>openssh (for multi-instance only)
        <li>horovod (for multi-instance only)
        <li>Activate the `tensorflow` conda environment
        <pre>conda activate tensorflow</pre>
      </ul>
    </td>
    <td>
      <p>To run without AI Kit you will need:</p>
      <ul>
        <li>Python 3
        <li>intel-tensorflow>=2.5.0
        <li>git
        <li>numactl
        <li>gcc-8 (for multi-instance only)
        <li>g++-8 (for multi-instance only)
        <li>libopenmpi-dev (for multi-instance only)
        <li>openmpi (for multi-instance only)
        <li>openssh (for multi-instance only)
        <li>horovod (for multi-instance only)
        <li>A clone of the Model Zoo repo<br />
        <pre>git clone https://github.com/IntelAI/models.git</pre>
      </ul>
    </td>
  </tr>
</table>

After you've completed the setup, export environment variables for the `DATASET_DIR`
and an `OUTPUT_DIR` where log files will be written, then run a
[quickstart script](#quick-start-scripts) from your model zoo directory.
```
# cd to your model zoo directory
cd models

# Set required environment variables
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>

# Run a quickstart script
./quickstart/language_translation/tensorflow/transformer_mlperf/training/cpu/fp32/<script name>.sh
```

<!--- 90. Resource Links-->
## Additional Resources

* To run more advanced use cases, see the instructions [here](Advanced.md)
  for calling the `launch_benchmark.py` script directly.
* To run the model using docker, please see the [oneContainer](http://software.intel.com/containers)
  workload container:<br />
  [https://software.intel.com/content/www/us/en/develop/articles/containers/transformer-lt-mlperf-fp32-training-tensorflow-container.html](https://software.intel.com/content/www/us/en/develop/articles/containers/transformer-lt-mlperf-fp32-training-tensorflow-container.html).

