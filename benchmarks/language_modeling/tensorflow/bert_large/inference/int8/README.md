<!--- 0. Title -->
# BERT Large Int8 inference

<!-- 10. Description -->

This document has instructions for running
[BERT](https://github.com/google-research/bert#what-is-bert) Int8 inference
using Intel-optimized TensorFlow.

<!--- 30. Datasets -->
## Datasets

### BERT Large Data
Download and unzip the BERT Large uncased (whole word masking) model from the
[google bert repo](https://github.com/google-research/bert#pre-trained-models).
Then, download the Stanford Question Answering Dataset (SQuAD) dataset file `dev-v1.1.json` into the `wwm_uncased_L-24_H-1024_A-16` directory that was just unzipped.

```
wget https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip
unzip wwm_uncased_L-24_H-1024_A-16.zip

wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -P wwm_uncased_L-24_H-1024_A-16
```
Set the `DATASET_DIR` to point to that directory when running BERT Large inference using the SQuAD data.

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`int8_batch_inference.sh`](/quickstart/language_modeling/tensorflow/bert_large/inference/cpu/int8/int8_batch_inference.sh) | Runs batch inference using a batch size of 32. |
| [`int8_online_inference.sh`](/quickstart/language_modeling/tensorflow/bert_large/inference/cpu/int8/int8_online_inference.sh) | Runs online inference using a batch size of 1. |
| [`int8_accuracy.sh`](/quickstart/language_modeling/tensorflow/bert_large/inference/cpu/int8/int8_accuracy.sh) | Run an accuracy test using a batch size of 32. |
| [`multi_instance_batch_inference.sh`](/quickstart/language_modeling/tensorflow/bert_large/inference/cpu/int8/multi_instance_batch_inference.sh) | A multi-instance run that uses all the cores for each socket for each instance with a batch size of 32. |
| [`multi_instance_online_inference.sh`](/quickstart/language_modeling/tensorflow/bert_large/inference/cpu/int8/multi_instance_online_inference.sh) | A multi-instance run that uses 4 cores for each instance with a batch size of 1. |

<!--- 50. Bare Metal -->
## Bare Metal

> If you are running using AI Kit, first follow the
> [instructions here](/docs/general/tensorflow/AIKit.md) to get your environment setup.

To run on bare metal, the following prerequisites must be installed in your enviornment.
If you are using AI Kit, your TensorFlow conda environment should already have Python and
TensorFlow installed.
* Python 3
* [intel-tensorflow==2.4.0](https://pypi.org/project/intel-tensorflow/)
* numactl
* git
* unzip
* wget

Clone the Model Zoo repo:
```
git clone https://github.com/IntelAI/models.git
```

Download and unzip the pretrained model. The path to this directory should
be set as the `CHECKPOINT_DIR` before running quickstart scripts.
```
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/bert_large_checkpoints.zip
unzip bert_large_checkpoints.zip
export CHECKPOINT_DIR=$(pwd)/bert_large_checkpoints
```

Download the frozen graph. The path to this file should be set in the
`PRETRAINED_MODEL` environment variable before running the model.
```
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_4_0/asymmetric_per_channel_bert_int8.pb
export PRETRAINED_MODEL=$(pwd)/asymmetric_per_channel_bert_int8.pb
```

Once the above dependencies have been installed, set environment variables,
and then run a quickstart script. See the [datasets](#datasets) and
[list of quickstart scripts](#quick-start-scripts)  for more details on
the different options.

The snippet below shows how to run a quickstart script:
```
# cd to your model zoo directory
cd models

export DATASET_DIR=<path to the SQuAD dataset>
export OUTPUT_DIR=<directory where log files will be saved>
export CHECKPOINT_DIR=<path to the pretrained model checkpoints>
export PRETRAINED_MODEL=<path to the frozen graph>

# Run a script for your desired usage
./quickstart/language_modeling/tensorflow/bert_large/inference/cpu/int8/<script name>.sh
```

<!--- 90. Resource Links-->
## Additional Resources

* To run more advanced use cases, see the instructions [here](Advanced.md)
  for calling the `launch_benchmark.py` script directly.

