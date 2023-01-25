# TensorFlow SSD-ResNet34 inference

## Description
This document has instructions for running SSD-ResNet34 inference using Intel-optimized TensorFlow.

## Pull Command
```
docker pull intel/object-detection:spr-ssd-resnet34-inference
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
         <td>Runs multi instance realtime inference (batch-size=1) using 4 cores per instance for the specified precision (fp32, int8 or bfloat16). Uses synthetic data with an input size of 1200x1200. Waits for all instances to complete, then prints a summarized throughput value.</td>
      </tr>
      <tr>
         <td>inference_throughput.sh</td>
         <td>Runs multi instance batch inference (batch-size=16) using 1 instance per socket for the specified precision (fp32, int8 or bfloat16). Uses synthetic data with an input size of 1200x1200. Waits for all instances to complete, then prints a summarized throughput value.</td>
      </tr>
      <tr>
         <td>accuracy.sh</td>
         <td>Measures the inference accuracy (providing a `DATASET_DIR` environment variable is required) for the specified precision (fp32, int8 or bfloat16).</td>
      </tr>
   </tbody>
</table>

## Datasets
The SSD-ResNet34 accuracy script `accuracy.sh` uses the
[COCO validation dataset](http://cocodataset.org) in the TF records
format. See the [COCO dataset document](https://github.com/IntelAI/models/tree/master/datasets/coco) for
instructions on downloading and preprocessing the COCO validation dataset.
The inference scripts use synthetic data, so no dataset is required.

After the script to convert the raw images to the TF records file completes, rename the tf_records file:
```
mv ${OUTPUT_DIR}/coco_val.record ${OUTPUT_DIR}/validation-00000-of-00001
```
Set the `DATASET_DIR` to the folder that has the `validation-00000-of-00001`
file when running the accuracy test. Note that the inference performance
test uses synthetic dataset.

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

To run Mask R-CNN inference, set environment variables to specify the dataset directory, precision and mode to run, and an output directory. 
```bash
# Set the required environment vars
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
export SCRIPT=<specify the script to run>
export PRECISION=<specify the precision to run>

docker run --rm \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env PRECISION=${PRECISION} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -it \
  --shm-size 8G \
  -w /workspace/tensorflow-spr-resnet34-inference \
  intel/object-detection:spr-ssd-resnet34-inference \
  /bin/bash quickstart/${SCRIPT}
```

## Documentation and Sources
#### Get Started​
[Docker* Repository](https://hub.docker.com/r/intel/object-detection)

[Main GitHub*](https://github.com/IntelAI/models)

[Release Notes](https://github.com/IntelAI/models/releases)

[Get Started Guide](https://github.com/IntelAI/models/blob/master/quickstart/object_detection/tensorflow/ssd-resnet34/inference/cpu/README_SPR_DEV_CAT.md)

#### Code Sources
[Dockerfile](https://github.com/IntelAI/models/tree/master/dockerfiles/tensorflow)

[Report Issue](https://community.intel.com/t5/Intel-Optimized-AI-Frameworks/bd-p/optimized-ai-frameworks)

## License Agreement
LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license](https://github.com/IntelAI/models/tree/master/third_party) file for additional details.

[View All Containers and Solutions 🡢](https://www.intel.com/content/www/us/en/developer/tools/software-catalog/containers.html?s=Newest)