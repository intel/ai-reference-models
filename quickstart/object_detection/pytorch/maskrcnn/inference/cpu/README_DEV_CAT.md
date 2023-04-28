# PyTorch Mask R-CNN inference

## Description
This document has instructions for running Mask R-CNN inference using Intel® Optimization for PyTorch*.

## Pull Command
```
docker pull intel/image-segmentation:spr-maskrcnn-inference
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
         <td>Runs multi instance realtime inference using 4 cores per instance for the specified precision (fp32, avx-fp32, bf16, or bf32) and mode (imperative or jit).</td>
      </tr>
      <tr>
         <td>inference_throughput.sh</td>
         <td>Runs multi instance batch inference using 24 cores per instance for the specified precision (fp32, avx-fp32, bf16, or bf32) and mode (imperative or jit).</td>
      </tr>
      <tr>
         <td>accuracy.sh</td>
         <td>Measures the inference accuracy for the specified precision (fp32, avx-fp32, bf16, or bf32) and mode (imperative or jit).</td>
      </tr>
   </tbody>
</table>

## Datasets
Download the 2017 [COCO dataset](https://cocodataset.org). Export the `DATASET_DIR` environment variable to specify the directory where the dataset will be downloaded. This environment variable will be used again when running quickstart scripts.

```bash
export DATASET_DIR=<directory where the dataset will be saved>

mkdir -p ${DATASET_DIR}/coco
curl -o ${DATASET_DIR}/coco/val2017.zip http://images.cocodataset.org/zips/val2017.zip
unzip ${DATASET_DIR}/coco/val2017.zip -d ${DATASET_DIR}/coco/
curl -o ${DATASET_DIR}/coco/annotations_trainval2017.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip ${DATASET_DIR}/coco/annotations_trainval2017.zip -d ${DATASET_DIR}/coco/
```

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

Download the pretrained model and set the `CHECKPOINT_DIR` environment variable to point to the directory where the weights file is downloaded.
```bash
curl -O https://download.pytorch.org/models/maskrcnn/e2e_mask_rcnn_R_50_FPN_1x.pth
export CHECKPOINT_DIR=$(pwd)
```

To run Mask R-CNN inference, set environment variables to specify the dataset directory, precision and mode to run, and an output directory. 
```bash
# Set the required environment vars
export MODE=<imperative or jit>
export CHECKPOINT_DIR=<path to the directory where the downloaded .pth file is located>
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
export SCRIPT=<specify the script to run>
export PRECISION=<specify the precision to run>

docker run --rm \
  --env CHECKPOINT_DIR=${CHECKPOINT_DIR} \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --volume ${CHECKPOINT_DIR}:${CHECKPOINT_DIR} \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -it \
  --shm-size 8G \
  -w /workspace/pytorch-spr-maskrcnn-inference \
  intel/image-segmentation:spr-maskrcnn-inference \
  /bin/bash quickstart/${SCRIPT} ${PRECISION} ${MODE}
```

## Documentation and Sources
#### Get Started​
[Docker* Repository](https://hub.docker.com/r/intel/image-segmentation)

[Main GitHub*](https://github.com/IntelAI/models)

[Release Notes](https://github.com/IntelAI/models/releases)

[Get Started Guide](https://github.com/IntelAI/models/blob/master/quickstart/object_detection/pytorch/maskrcnn/inference/cpu/README.md)

#### Code Sources
[Dockerfile](https://github.com/IntelAI/models/tree/master/dockerfiles/pytorch)

[Report Issue](https://community.intel.com/t5/Intel-Optimized-AI-Frameworks/bd-p/optimized-ai-frameworks)

## License Agreement
LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party software included with the Software Package. Please refer to the [license](https://github.com/IntelAI/models/tree/master/third_party) file for additional details.

[View All Containers and Solutions 🡢](https://www.intel.com/content/www/us/en/developer/tools/software-catalog/containers.html?s=Newest)
