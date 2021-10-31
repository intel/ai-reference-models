#!/usr/bin/env bash
#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Intel Corporation
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
# SPDX-License-Identifier: EPL-2.0
#

########## Variables to be defined - run it in research folder

SPLIT=faster-rcnn1 #change to your favorite name

FROZEN_GRAPH=$1
TF_RECORD_FILES=$2
TF_MODELS_ROOT=$3

if [[ -z ${TF_MODELS_ROOT} ]] || [[ ! -d ${TF_MODELS_ROOT} ]]; then
  echo "You must specify the root of the tensorflow/models source tree in the TF_MODELS_ROOT environment variable."
  exit 1
fi

export PYTHONPATH=$PYTHONPATH:${TF_MODELS_ROOT}/research:${TF_MODELS_ROOT}/research/slim:${TF_MODELS_ROOT}/research/object_detection

python object_detection/inference/infer_detections.py \
  --input_tfrecord_paths=$TF_RECORD_FILES \
  --output_tfrecord_path=${SPLIT}_detections.tfrecord \
  --inference_graph=$FROZEN_GRAPH \
  --discard_image_pixels=True


mkdir -p ${SPLIT}_eval_metrics

echo "
label_map_path: '${TF_MODELS_ROOT}/research/object_detection/data/mscoco_label_map.pbtxt'
tf_record_input_reader: { input_path: '${SPLIT}_detections.tfrecord' }
" > ${SPLIT}_eval_metrics/${SPLIT}_input_config.pbtxt

echo "
metrics_set: 'coco_detection_metrics'
" > ${SPLIT}_eval_metrics/${SPLIT}_eval_config.pbtxt


python object_detection/metrics/offline_eval_map_corloc.py \
  --eval_dir=${SPLIT}_eval_metrics \
  --eval_config_path=${SPLIT}_eval_metrics/${SPLIT}_eval_config.pbtxt \
  --input_config_path=${SPLIT}_eval_metrics/${SPLIT}_input_config.pbtxt
