# TensorFlow MLPerf 3D U-Net inference

## Description
This document has instructions for running MLPerf 3D U-Net inference using
Intel-optimized TensorFlow.

## Pull Command
```
docker pull intel/image-segmentation:centos-tf-cpu-3d-unet-mlperf-inference
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
         <td>Runs multi instance realtime inference using 4 cores per instance for the specified precision (int8, fp32 or bfloat16) with 100 steps and 50 warmup steps. Dummy data is used for performance evaluation. Waits for all instances to complete, then prints a summarized throughput value.</td>
      </tr>
      <tr>
         <td>inference_throughput.sh</td>
         <td>Runs multi instance batch inference using 1 instance per socket for the specified precision (int8, fp32 or bfloat16) with 100 steps and 50 warmup steps. Dummy data is used for performance evaluation. Waits for all instances to complete, then prints a summarized throughput value.</td>
      </tr>
      <tr>
         <td>accuracy.sh</td>
         <td>Measures the inference accuracy (providing a `DATASET_DIR` environment variable is required) for the specified precision (int8, fp32 or bfloat16).</td>
      </tr>
   </tbody>
</table>

## Datasets
Download [Brats 2019](https://www.med.upenn.edu/cbica/brats2019/data.html) separately and unzip the dataset.

Set the `DATASET_DIR` to point to the directory that contains the dataset files when running MLPerf 3D U-Net accuracy script.

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

To run MLPerf 3D U-Net inference, set environment variables to specify the dataset directory, precision and mode to run, and an output directory. 
```bash
# Set the required environment vars
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
export SCRIPT=<specify the script to run>
export PRECISION=<specify the precision to run>

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
  -w /workspace/tf-3d-unet-mlperf-inference \
  intel/image-segmentation:centos-tf-cpu-3d-unet-mlperf-inference \
  /bin/bash quickstart/${SCRIPT}
```

## Documentation and Sources
#### Get Started​
[Docker* Repository](https://hub.docker.com/r/intel/image-segmentation)

[Main GitHub*](https://github.com/IntelAI/models)

[Release Notes](https://github.com/IntelAI/models/releases)

[Get Started Guide](https://github.com/IntelAI/models/blob/master/quickstart/image-segmentation/tensorflow/3d_unet_mlperf/inference/cpu/README_DEV_CAT.md)

#### Code Sources
[Dockerfile](https://github.com/IntelAI/models/tree/master/dockerfiles/tensorflow)

[Report Issue](https://community.intel.com/t5/Intel-Optimized-AI-Frameworks/bd-p/optimized-ai-frameworks)

## License Agreement
LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license](https://github.com/IntelAI/models/tree/master/third_party) file for additional details.

[View All Containers and Solutions 🡢](https://www.intel.com/content/www/us/en/developer/tools/software-catalog/containers.html?s=Newest)
