#!/usr/bin/env bash
#
# Copyright (c) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

MODEL_DIR=${MODEL_DIR-$PWD}

echo 'MODEL_DIR='$MODEL_DIR
echo 'OUTPUT_DIR='$OUTPUT_DIR
echo 'DATASET_DIR='$DATASET_DIR

if [[ -z ${OUTPUT_DIR} ]]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

if [[ -z ${DATASET_DIR} ]]; then
  echo "The required environment variable DATASET_DIR has not been set"
  exit 1
fi

if [[ ! -d ${DATASET_DIR} ]]; then
  echo "The DATASET_DIR '${DATASET_DIR}' does not exist"
  exit 1
fi

if [[ -z ${TF_MODELS_DIR} ]]; then
  echo "The required environment variable TF_MODELS_DIR has not been set"
  exit 1
fi

if [[ ! -d ${TF_MODELS_DIR} ]]; then
  echo "The TF_MODELS_DIR '${TF_MODELS_DIR}' does not exist"
  exit 1
fi

# If a path to the pretrained model dir was not provided, unzip the pretrained
# model from the model package
if [[ -z ${PRETRAINED_MODEL} ]]; then
  tar -xvf ${MODEL_DIR}/faster_rcnn_resnet50_fp32_coco_pretrained_model.tar.gz
  PRETRAINED_MODEL=${MODEL_DIR}/faster_rcnn_resnet50_fp32_coco
fi

if [[ ! ${PRETRAINED_MODEL} ]]; then
  echo "The pretrained model directory (${PRETRAINED_MODEL}) does not exist."
  exit 1
fi

# Replace paths in the pipeline config file
line_128=$(sed -n '128p' ${PRETRAINED_MODEL}/pipeline.config)
new_line_128="  label_map_path: \"$PRETRAINED_MODEL/mscoco_label_map.pbtxt\""
sed -i.bak "128s+$line_128+$new_line_128+" ${PRETRAINED_MODEL}/pipeline.config
line_132=$(sed -n '132p' ${PRETRAINED_MODEL}/pipeline.config)
new_line_132="    input_path: \"$DATASET_DIR/coco_val.record\""
sed -i.bak "132s+$line_132+$new_line_132+" ${PRETRAINED_MODEL}/pipeline.config

source "${MODEL_DIR}/quickstart/common/utils.sh"
_command python ${MODEL_DIR}/benchmarks/launch_benchmark.py \
  --model-name faster_rcnn \
  --mode inference \
  --precision fp32 \
  --framework tensorflow \
  --model-source-dir ${TF_MODELS_DIR} \
  --output-dir ${OUTPUT_DIR} \
  --data-location ${DATASET_DIR} \
  --in-graph ${PRETRAINED_MODEL}/frozen_inference_graph.pb \
  --accuracy-only \
  $@
