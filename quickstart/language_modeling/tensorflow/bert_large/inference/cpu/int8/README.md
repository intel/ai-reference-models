<!--- 0. Title -->
# BERT Large Int8 inference

<!-- 10. Description -->

This document has instructions for running
[BERT](https://github.com/google-research/bert#what-is-bert) Int8 inference
using Intel-optimized TensorFlow.

<!--- 20. Download link -->
## Download link

[bert-large-int8-inference.tar.gz](https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/bert-large-int8-inference.tar.gz)

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

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3
* [intel-tensorflow>=2.5.0](https://pypi.org/project/intel-tensorflow/)
* numactl
* git
* unzip

Once the above dependencies have been installed, download and untar the model
package, set environment variables, and then run a quickstart script. See the
[datasets](#datasets) and [list of quickstart scripts](#quick-start-scripts) 
for more details on the different options.

The snippet below shows how to run a quickstart script:
```
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/bert-large-int8-inference.tar.gz
tar -xvf bert-large-int8-inference.tar.gz
cd bert-large-int8-inference

DATASET_DIR=<path to the SQuAD dataset>
OUTPUT_DIR=<directory where log files will be saved>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

# Run a script for your desired usage
./quickstart/<script name>.sh
```

<!-- 60. Docker -->
## Docker

The BERT Large Int8 inference model container includes the scripts,
pretrained model, and dependencies needed to run BERT Large Int8
inference. To run one of the quickstart scripts using this container, you'll
need to provide volume mounts for the dataset and an output directory
where log files will be written.

The snippet below shows how to run a quickstart script:
```
DATASET_DIR=<path to the SQuAD dataset>
OUTPUT_DIR=<directory where log files will be saved>
# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env BATCH_SIZE=${BATCH_SIZE} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -it \
  intel/language-modeling:tf-latest-bert-large-int8-inference \
  /bin/bash ./quickstart/<script name>.sh
```

If you are new to docker and are running into issues with the container,
see [this document](https://github.com/IntelAI/models/tree/master/docs/general/docker.md)
for troubleshooting tips.

<!--- 80. License -->
## License

[LICENSE](/LICENSE)

