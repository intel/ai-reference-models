#!/bin/bash
# Copyright (c) 2023 Intel Corporation
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
# ============================================================================

set -x

: ${DATA_DIR:=${1:${PWD}/data}}
: ${MODEL_DIR:=${2:${PWD}/model}}

CAL_SAMPLES=(128)
GROUPSIZE=(-1)
COMPRESSION_FACTOR=2

for cal_size in ${CAL_SAMPLES[@]}; do
  for g in ${GROUPSIZE[@]}; do
    echo "Running groups ${g} and samples ${cal_size}"
    python -u gptj.py --model ${MODEL_DIR} \
    --wbits 4 \
    --true-sequential \
    --act-order \
    --groupsize ${g} \
    --save ${MODEL_DIR}/gpt-j-quantized_model_${g}g_${cal_size}samples.pt \
    --calib-data-path ${DATA_DIR}/cnn_dailymail_calibration.json \
    --nsamples ${cal_size} \
    --quant-config-output ${MODEL_DIR}/gpt-j-quantized_model_params.json \
    --compression-factor ${COMPRESSION_FACTOR} \
    --compression-dim "N" \
    --calib-iters ${cal_size} \
    --quantize-lm-head \
    2>&1 | tee log_${g}groups_${cal_size}samples_cf_${COMPRESSION_FACTOR}.log
  done
done

mv ${MODEL_DIR}/gpt-j-quantized_model_${g}g_${cal_size}samples.pt ${MODEL_DIR}/int4_weight.pt

set +x
