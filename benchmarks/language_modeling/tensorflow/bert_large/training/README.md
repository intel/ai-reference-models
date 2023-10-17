<!--- 0. Title -->
# TensorFlow BERT Large Training

<!-- 10. Description -->

This section has instructions for running BERT Large Training with the SQuAD dataset.

Set the `OUTPUT_DIR` to point to the directory where all logs will get stored.
Set the `PRECISION` to choose the appropriate precision for training. Choose from `fp32`, `bfloat16`, or `fp16`

## Datasets

### SQuAD data
Download and unzip the BERT Large uncased (whole word masking) model from the
[google bert repo](https://github.com/google-research/bert#pre-trained-models).
Set the `DATASET_DIR` to point to this directory when running BERT Large.
```
mkdir -p $DATASET_DIR && cd $DATASET_DIR
wget https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip
unzip wwm_uncased_L-24_H-1024_A-16.zip

wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -P wwm_uncased_L-24_H-1024_A-16
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -P wwm_uncased_L-24_H-1024_A-16
```

## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `training_squad.sh` | Uses mpirun to execute 1 process per socket for BERT Large training with the specified precision (fp32, bfloat16 or fp16). Logs for each instance are saved to the output directory. |


# TensorFlow BERT Large Pretraining

<!-- 10. Description -->

This section has instructions for running BERT Large Pretraining
using Intel-optimized TensorFlow.


<!--- 30. Datasets -->
## Datasets

### SQuAD data
Download and unzip the BERT Large uncased (whole word masking) model from the
[google bert repo](https://github.com/google-research/bert#pre-trained-models).
Set the `DATASET_DIR` to point to this directory when running BERT Large.
```
mkdir -p $DATASET_DIR && cd $DATASET_DIR
wget https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip
unzip wwm_uncased_L-24_H-1024_A-16.zip

wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -P wwm_uncased_L-24_H-1024_A-16
```

Follow [instructions to generate BERT pre-training dataset](https://github.com/IntelAI/models/blob/bert-lamb-pretraining-tf-2.2/quickstart/language_modeling/tensorflow/bert_large/training/bfloat16/HowToGenerateBERTPretrainingDataset.txt)
in TensorFlow record file format. The output TensorFlow record files are expected to be located in the dataset directory `${DATASET_DIR}/tf_records`. An example for the TF record file path should be
`${DATASET_DIR}/tf_records/part-00430-of-00500`.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `pretraining.sh` | Uses mpirun to execute 1 process per socket for BERT Large pretraining with the specified precision (fp32, bfloat16 or fp16). Logs for each instance are saved to the output directory. |

<!--- 50. Baremetal -->
## Run the model

### Setup on baremetal

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
        <li>numactl
        <li>unzip
        <li>wget
        <li>openmpi-bin (only required for multi-instance)
        <li>openmpi-common (only required for multi-instance)
        <li>openssh-client (only required for multi-instance)
        <li>openssh-server (only required for multi-instance)
        <li>libopenmpi-dev (only required for multi-instance)
        <li>horovod==0.27.0 (only required for multi-instance)
        <li>Activate the `tensorflow` conda environment
        <pre>conda activate tensorflow</pre>
      </ul>
    </td>
    <td>
      <p>To run without AI Tools you will need:</p>
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
        <li>horovod==0.27.0 (only required for multi-instance)
        <li>A clone of the AI Reference Models repo<br />
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
* Use `--amp` flag to run with grappler Auto-Mixed Precision pass with FP16 precision: `./quickstart/language_modeling/tensorflow/bert_large/inference/cpu/<script_name.sh> --amp`. By default, BERT-Large-SQuAD FP16 inference runs with Keras Mixed-Precision policy. For more information on grappler Auto-Mixed Precision, please see [https://www.intel.com/content/www/us/en/developer/articles/guide/getting-started-with-automixedprecisionmkl.html](https://www.intel.com/content/www/us/en/developer/articles/guide/getting-started-with-automixedprecisionmkl.html).

### Run on Linux

Set environment variables to
specify the dataset directory, precision to run, and
an output directory.
```
# Navigate to the container package directory
cd models

# Set the required environment vars
export PRECISION=<specify the precision to run:fp32, bfloat16 or fp16>
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
export CHECKPOINT_DIR=<path to the downloaded checkpoints folder>

# Run the container with pretraining.sh quickstart script
./quickstart/language_modeling/tensorflow/bert_large/training/cpu/pretraining.sh
```

<!--- 90. Resource Links-->
## Additional Resources

* To run more advanced use cases, see the instructions for the available precisions [FP32](fp32/Advanced.md) [<int8 precision>](<int8 advanced readme link>) [BFloat16](bfloat16/Advanced.md) [FP16](fp16/Advanced.md) for calling the `launch_benchmark.py` script directly.
* To run the model using docker, please see the [Intel® Developer Catalog](https://www.intel.com/content/www/us/en/developer/tools/software-catalog/containers.html)
  workload container:<br />
  [https://www.intel.com/content/www/us/en/developer/articles/containers/bert-large-fp32-training-tensorflow-container.html](https://www.intel.com/content/www/us/en/developer/articles/containers/bert-large-fp32-training-tensorflow-container.html).
