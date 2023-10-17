#!/usr/bin/env bash
#
# Copyright (c) 2021-2023 Intel Corporation
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
BATCH_SIZE=${BATCH_SIZE-64}

if [[ -z "${DATASET_DIR}" ]]; then
  echo "The required environment variable DATASET_DIR has not been set"
  exit 1
fi

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "The DATASET_DIR '${DATASET_DIR}' does not exist"
  exit 1
fi

if [[ -z "${BERT_WEIGHT}" ]]; then
  echo "The required environment variable BERT_WEIGHT has not been set"
  exit 1
fi

if [[ ! -d "${BERT_WEIGHT}" ]]; then
  echo "The DATASET_DIR '${BERT_WEIGHT}' does not exist"
  exit 1
fi

if [[ -z $OUTPUT_DIR ]]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

if [[ -z $PRECISION ]]; then
  echo "The required environment variable PRECISION has not been set"
  exit 1
fi

if [[ -z $NUM_OAM ]]; then
  echo "The required environment variable NUM_OAM has not been set."
  exit 1
fi
# Create the output directory, if it doesn't already exist
mkdir -p $OUTPUT_DIR

if [[ ${PRECISION} != "FP16" ]]; then
  echo "BERT Large Inference currently supports FP16 precision"
  exit 1
fi

if [[ ${NUM_OAM} == "4" ]]; then
  echo "bertsquad fp16 inference plain nchw on ${NUM_OAM} OAM modules"
  NUM_TILES_PER_GPU=2
  cd ${MODEL_DIR}/models/language_modeling/pytorch/bert_large/inference/gpu/
  for i in $( eval echo {0..$((NUM_OAM-1))} )
    do
        for j in $( eval echo {0..$((NUM_TILES_PER_GPU-1))} )
            do
              str+=("ZE_AFFINITY_MASK="${i}"."${j}" bash cmd_infer.sh \
                    -m bert_large \
                    -d xpu \
                    -b $BATCH_SIZE \
                    -t ${PRECISION} \
                    -o None 2>&1 | tee ${OUTPUT_DIR}/bertsquad_fp16_inf_plain_nchw_c${i}_t${j}_raw.log & ")

            done
      done
        
  #     bertsquad_log_analysis ${OUTPUT_DIR}/bertsquad_fp16_inf_plain_nchw_t0_raw.log ${OUTPUT_DIR}/bertsquad_fp16_inf_plain_nchw_t0.log inference ${BATCH_SIZE}
  # bertsquad_log_analysis ${OUTPUT_DIR}/bertsquad_fp16_inf_plain_nchw_t1_raw.log ${OUTPUT_DIR}/bertsquad_fp16_inf_plain_nchw_t1.log inference ${BATCH_SIZE}
  str=${str[@]}
  cmd_line=${str::-2}
  eval $cmd_line
  wait
  cd -
else
    echo "Currently only x4 OAM Modules are supported"
    exit 1
fi
