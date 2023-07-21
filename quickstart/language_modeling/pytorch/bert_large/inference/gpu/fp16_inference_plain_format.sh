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

#source ${MODEL_DIR}/quickstart/setvars.sh

if [[ -z "${Tile}" ]]; then
    Tile=${Tile-1}
else
    Tile=${Tile}
fi

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

# Create the output directory, if it doesn't already exist
mkdir -p $OUTPUT_DIR

bertsquad_log_analysis() {
    # $1 : src raw log
    # $2 : dst format log
    # $3 : inference or training
    # $4 : bs

    if [ -f $2 ]; then
        rm $2
    fi

    bs=$4

    if [ "inference" == "$3" ]; then
        echo -e 'Batch Size: ' $bs >$2
        cat $1 | grep latency | tail -n6 | head -n4 |
            awk -v bs=${bs} -F ' ' '{sum+=$8} END{printf "Performance Benchmark Time: %.3f sec, Throughput: %.2f seq/sec\n", sum/4, bs*4/sum}' >>$2
        grep "\"f1\": " $1 | awk -F ' ' '{printf "Accuracy: f1 %.4f\n", $NF}' >>$2
    elif [ "training" == "$3" ]; then
        # only for fine tune (accuracy only)
        echo -e 'Batch Size: ' $bs >$2
        echo -e 'Performance Benchmark Time: N/A' >>$2
        grep "\"f1\": " $1 | awk -F ' ' '{printf "Accuracy: f1 %.4f\n", $NF}' >>$2
    else
        echo -e 'Invalid input! Only inference or training are supported.'
        exit 0
    fi
}

if [[ ${Tile} == "1" ]]; then
  echo "bertsquad fp16 inference plain nchw"
  cd ${MODEL_DIR}/models/language_modeling/pytorch/bert_large/inference/gpu/
  bash cmd_infer.sh \
      -m bert_large \
      -d xpu \
      -b $BATCH_SIZE \
      -t FP16 \
      -o None 2>&1 | tee ${OUTPUT_DIR}/bertsquad_fp16_inf_plain_nchw_t0_raw.log
  wait
  bertsquad_log_analysis ${OUTPUT_DIR}/bertsquad_fp16_inf_plain_nchw_t0_raw.log ${OUTPUT_DIR}/bertsquad_fp16_inf_plain_nchw_t0.log inference ${BATCH_SIZE}
  cd -
elif [[ ${Tile} == "2" ]]; then
  echo "bertsquad fp16 inference plain nchw 2 tile"
  cd ${MODEL_DIR}/models/language_modeling/pytorch/bert_large/inference/gpu/
  ZE_AFFINITY_MASK=0.0 bash cmd_infer.sh \
      -m bert_large \
      -d xpu \
      -b $BATCH_SIZE \
      -t FP16 \
      -o None 2>&1 | tee ${OUTPUT_DIR}/bertsquad_fp16_inf_plain_nchw_t0_raw.log &
  ZE_AFFINITY_MASK=0.1 bash cmd_infer.sh \
      -m bert_large \
      -d xpu \
      -b $BATCH_SIZE \
      -t FP16 \
      -o None 2>&1 | tee ${OUTPUT_DIR}/bertsquad_fp16_inf_plain_nchw_t1_raw.log
  wait
  bertsquad_log_analysis ${OUTPUT_DIR}/bertsquad_fp16_inf_plain_nchw_t0_raw.log ${OUTPUT_DIR}/bertsquad_fp16_inf_plain_nchw_t0.log inference ${BATCH_SIZE}
  bertsquad_log_analysis ${OUTPUT_DIR}/bertsquad_fp16_inf_plain_nchw_t1_raw.log ${OUTPUT_DIR}/bertsquad_fp16_inf_plain_nchw_t1.log inference ${BATCH_SIZE}
  cd -
else
    echo "The specified Tile '${Tile}' is unsupported."
    echo "Supported tile number are: 1 and 2"
    exit 1
fi
