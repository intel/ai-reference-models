#!/usr/bin/env bash
#
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
#

MODEL_DIR=${MODEL_DIR-$PWD}
BATCH_SIZE=${BATCH_SIZE-1}
PRECISION=${PRECISION-fp32}

if [[ -z $OUTPUT_DIR ]]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

echo "Stable Diffusion Inference Inference"
if [[ ${PRECISION} == "fp32" ]]; then

python -u ${MODEL_DIR}/models/generative-ai/pytorch/stable_diffusion/inference/gpu/main.py \
    --save_image --channels_last 2>&1 | tee $OUTPUT_DIR/${PRECISION}_stable_diffusion_logs.txt

elif [[ ${PRECISION} == "fp16" ]]; then

python -u ${MODEL_DIR}/models/generative-ai/pytorch/stable_diffusion/inference/gpu/main.py \
    --save_image --channels_last --precision fp16 2>&1 | tee $OUTPUT_DIR/${PRECISION}_stable_diffusion_logs.txt
else
  echo "Stable Diffusion currently supports fp32 and fp16 precisions."
  exit 1
fi
