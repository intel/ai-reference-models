# TensorFlow BERT Large Pretraining

## Description
This document has instructions for running BERT Large Pretraining using Intel-optimized TensorFlow.

## Pull Command
```
docker pull intel/language-modeling:centos-tf-cpu-bert-large-pretraining
```

<table>
   <thead>
      <tr>
         <th>Script name</th>
         <th>Description</th>
      </tr>
   </thead>
   <tbody>
      <tr>
         <td>pretraining.sh</td>
         <td>Uses mpirun to execute 2 processes with 1 process per socket for BERT Large pretraining with the specified precision (fp32 or bfloat16). Logs for each instance are saved to the output directory.</td>
      </tr>
   </tbody>
</table>

## Datasets
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

## Docker Run
(Optional) Export related proxy into docker environment.
```bash
export DOCKER_RUN_ENVS="-e ftp_proxy=${ftp_proxy} \
  -e FTP_PROXY=${FTP_PROXY} -e http_proxy=${http_proxy} \
  -e HTTP_PROXY=${HTTP_PROXY} -e https_proxy=${https_proxy} \
  -e HTTPS_PROXY=${HTTPS_PROXY} -e no_proxy=${no_proxy} \
  -e NO_PROXY=${NO_PROXY} -e socks_proxy=${socks_proxy} \
  -e SOCKS_PROXY=${SOCKS_PROXY}"
```

To run BERT Large Pretraining, set environment variables to specify the dataset directory, precision to run, and an output directory.
```bash
# Set the required environment vars
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
export PRECISION=<specify the precision to run (fp32 or bfloat16)>

# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

docker run --rm \
  --env BATCH_SIZE=${BATCH_SIZE} \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env PRECISION=${PRECISION} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  ${DOCKER_RUN_ENVS} \
  --privileged --init -it \
  --shm-size 8G \
  -w /workspace/tf-spr-bert-large-pretraining \
  intel/language-modeling:centos-tf-cpu-bert-large-pretraining \
  /bin/bash models_v2/pretraining.sh
```

## Documentation and Sources
#### Get Started​
[Docker* Repository](https://hub.docker.com/r/intel/language-modeling)

[Main GitHub*](https://github.com/IntelAI/models)

[Release Notes](https://github.com/IntelAI/models/releases)

[Get Started Guide](https://github.com/IntelAI/models/blob/master/quickstart/language_modeling/tensorflow/bert_large/training/cpu/README_DEV_CAT.md)

#### Code Sources
[Dockerfile](https://github.com/IntelAI/models/tree/master/dockerfiles/tensorflow)

[Report Issue](https://community.intel.com/t5/Intel-Optimized-AI-Frameworks/bd-p/optimized-ai-frameworks)

## License Agreement
LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license](https://github.com/IntelAI/models/tree/master/third_party) file for additional details.

[View All Containers and Solutions 🡢](https://www.intel.com/content/www/us/en/developer/tools/software-catalog/containers.html?s=Newest)
