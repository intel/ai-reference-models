# TensorFlow BERT Large inference

## Description
This document has instructions for running BERT Large inference using
Intel-optimized TensorFlow.

## Pull Command
```
docker pull intel/language-modeling:centos-tf-cpu-bert-large-inference
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
         <td>inference_realtime.sh</td>
         <td>Runs multi instance realtime inference for BERT large (SQuAD) using 4 cores per instance with batch size 1 ( for precisions: fp32, int8, bfloat16 and bfloat32) to compute latency. Waits for all instances to complete, then prints a summarized throughput value.</td>
      </tr>
      <tr>
        <td>inference_realtime_weight_sharing.sh</td>
        <td>Runs multi instance realtime inference with weight sharing for BERT large (SQuAD) using 4 cores per instance with batch size 1 ( for precisions: fp32, int8, bfloat16 and bfloat32) to compute latency for weight sharing. Waits for all instances to complete, then prints a summarized throughput value.</td>
      </tr>
      <tr>
         <td>inference_throughput.sh</td>
         <td>Runs multi instance batch inference for BERT large (SQuAD) using 1 instance per socket with batch size 128 (for precisions: fp32, int8 or bfloat16) to compute throughput. Waits for all instances to complete, then prints a summarized throughput value.</td>
      </tr>
      <tr>
         <td>accuracy.sh</td>
         <td>Measures BERT large (SQuAD) inference accuracy for the specified precision (fp32, int8 or bfloat16 and bfloat32).</td>
      </tr>
   </tbody>
</table>

## Datasets
Download and unzip the BERT Large uncased (whole word masking) model from the
[google bert repo](https://github.com/google-research/bert#pre-trained-models).
Then, download the Stanford Question Answering Dataset (SQuAD) dataset file `dev-v1.1.json` into the `wwm_uncased_L-24_H-1024_A-16` directory that was just unzipped.

```
wget https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip
unzip wwm_uncased_L-24_H-1024_A-16.zip

wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -P wwm_uncased_L-24_H-1024_A-16
```
Set the `DATASET_DIR` to point to that directory when running BERT Large inference using the SQuAD data.

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

To run BERT Large inference, set environment variables to specify the dataset directory, precision and mode to run, and an output directory.
```bash
# Set the required environment vars
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
export SCRIPT=<specify the script to run>
export PRECISION=<specify the precision to run>
export CORES_PER_INSTANCE=<specify the cores per instance. Otherwise, script assigns a default value>
export OMP_NUM_THREADS=<specify the number for OMP_NUM_THREADS. Otherwise, script assigns a default value>

# For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
export BATCH_SIZE=<customized batch size value>

docker run --rm \
  --env BATCH_SIZE=${BATCH_SIZE} \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env PRECISION=${PRECISION} \
  --env CORES_PER_INSTANCE=${CORES_PER_INSTANCE} \
  --env OMP_NUM_THREADS=${OMP_NUM_THREADS} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -it \
  --shm-size 8G \
  -w /workspace/tf-bert-large-inference \
  ${DOCKER_RUN_ENVS} \
  intel/language-modeling:centos-tf-cpu-bert-large-inference \
  /bin/bash models_v2/${SCRIPT}
```

## Documentation and Sources
#### Get Started​
[Docker* Repository](https://hub.docker.com/r/intel/language-modeling)

[Main GitHub*](https://github.com/IntelAI/models)

[Release Notes](https://github.com/IntelAI/models/releases)

[Get Started Guide](https://github.com/IntelAI/models/blob/master/quickstart/language_modeling/tensorflow/bert_large/inference/cpu/README_DEV_CAT.md)

#### Code Sources
[Dockerfile](https://github.com/IntelAI/models/tree/master/dockerfiles/tensorflow)

[Report Issue](https://community.intel.com/t5/Intel-Optimized-AI-Frameworks/bd-p/optimized-ai-frameworks)

## License Agreement
LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license](https://github.com/IntelAI/models/tree/master/third_party) file for additional details.

[View All Containers and Solutions 🡢](https://www.intel.com/content/www/us/en/developer/tools/software-catalog/containers.html?s=Newest)
