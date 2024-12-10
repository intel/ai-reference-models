# TensorFlow ResNet50 v1.5 training

## Description
This document has instructions for running ResNet50 v1.5 training using Intel-optimized TensorFlow.

## Pull Command
```
docker pull intel/image-recognition:centos-tf-cpu-resnet50v1-5-training
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
         <td>training.sh</td>
         <td>Uses mpirun to execute 1 processes with 1 process per socket with a batch size of 1024 for the specified precision (fp32, bfloat16 or fp16). Checkpoint files and logs for each instance are saved to the output directory.</td>
      </tr>
   </tbody>
</table>

Note that the ImageNet dataset is used in these ResNet50 v1.5 examples.

## Datasets

Download and preprocess the ImageNet dataset using the [instructions here](https://github.com/IntelAI/models/blob/master/datasets/imagenet/README.md).
After running the conversion script you should have a directory with the
ImageNet dataset in the TF records format.

Set the `DATASET_DIR` to point to this directory when running ResNet50 v1.5.

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

To run ResNet50 v1.5 training, set environment variables to specify the dataset directory, precision to run, and an output directory.
```bash
# Set the required environment vars
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
export PRECISION=<specify the precision to run (fp32 or bfloat16 or bfloat32 or fp16)>

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
  -w /workspace/tf-resnet50v1-5-training \
  intel/image-recognition:centos-tf-cpu-resnet50v1-5-training \
  /bin/bash models_v2/training_full.sh
```

## Documentation and Sources
#### Get Started​
[Docker* Repository](https://hub.docker.com/r/intel/image-recognition)

[Main GitHub*](https://github.com/IntelAI/models)

[Release Notes](https://github.com/IntelAI/models/releases)

[Get Started Guide](https://github.com/IntelAI/models/blob/master/quickstart/image_recognition/tensorflow/resnet50v1_5/training/cpu/README_DEV_CAT.md)

#### Code Sources
[Dockerfile](https://github.com/IntelAI/models/tree/master/dockerfiles/tensorflow)

[Report Issue](https://community.intel.com/t5/Intel-Optimized-AI-Frameworks/bd-p/optimized-ai-frameworks)

## License Agreement
LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license](https://github.com/IntelAI/models/tree/master/third_party) file for additional details.

[View All Containers and Solutions 🡢](https://www.intel.com/content/www/us/en/developer/tools/software-catalog/containers.html?s=Newest)
