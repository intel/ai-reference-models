<!--- 50. Baremetal -->
## Run the model

### Setup on baremetal

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
        <li>unzip
        <li>wget
        <li>openmpi-bin (only required for multi-instance)
        <li>openmpi-common (only required for multi-instance)
        <li>openssh-client (only required for multi-instance)
        <li>openssh-server (only required for multi-instance)
        <li>libopenmpi-dev (only required for multi-instance)
        <li>horovod==0.25.0 (only required for multi-instance)
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
        <li>openmpi-bin (only required for multi-instance)
        <li>openmpi-common (only required for multi-instance)
        <li>openssh-client (only required for multi-instance)
        <li>openssh-server (only required for multi-instance)
        <li>libopenmpi-dev (only required for multi-instance)
        <li>horovod==0.25.0 (only required for multi-instance)
        <li>A clone of the Model Zoo repo<br />
        <pre>git clone https://github.com/IntelAI/models.git</pre>
      </ul>
    </td>
  </tr>
</table>

### Download checkpoints:
```bash
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/bert_large_checkpoints.zip
unzip bert_large_checkpoints.zip
export CHECKPOINT_DIR=$(pwd)/bert_large_checkpoints
```

### Run on Linux

Set environment variables to
specify the dataset directory, precision to run, and
an output directory.
```
# Navigate to the container package directory
cd models

# Set the required environment vars
export PRECISION=<specify the precision to run:fp32 and bfloat16>
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
export CHECKPOINT_DIR=<path to the downloaded checkpoints folder>

# Run the container with pretraining.sh quickstart script
./quickstart/language_modeling/tensorflow/bert_large/training/cpu/pretraining.sh
```
