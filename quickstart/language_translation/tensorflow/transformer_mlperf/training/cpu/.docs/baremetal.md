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
[quickstart script](#quick-start-scripts) from your model zoo directory. To run
multi-instance training, set the `MPI_NUM_PROCESSES` var to specify the number of
instances to run (for example `export MPI_NUM_PROCESSES=2` for 2 instances).
```
# cd to your model zoo directory
cd models

# Set required environment variables
export DATASET_DIR=<path to the dataset>
export PRECISION=<set precision to fp32 or bfloat16>
export OUTPUT_DIR=<path to the directory where log files will be written>

# Optional: Specify the number of sockets to use for multi-instance training
export MPI_NUM_PROCESSES=<number of sockets to use>

# Run a quickstart script
./quickstart/language_translation/tensorflow/transformer_mlperf/training/cpu/<script name>.sh
```
