#!/usr/bin/env bash
#
# Copyright (c) 2021 Intel Corporation
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
BATCH_SIZE=${BATCH_SIZE-1024}
NUM_ITERATIONS=${NUM_ITERATIONS-10}



source ${MODEL_DIR}/quickstart/setvars.sh

if [[ -z "${DATASET_DIR}" ]]; then
  echo "The required environment variable DATASET_DIR has not been set"
  exit 1
fi

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "The DATASET_DIR '${DATASET_DIR}' does not exist"
  exit 1
fi

if [[ -z $OUTPUT_DIR ]]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

# Create the output directory, if it doesn't already exist
mkdir -p $OUTPUT_DIR

if [[ -z "${PRECISION}" ]]; then
  echo "The required environment variable PRECISION has not been set."
  echo "Please set PRECISION to fp32, fp16, or bf16."
  exit 1
fi

resnet50_log_analysis() {
    # $1 : src raw log
    # $2 : dst format log
    # $3 : inference or training
    # $4 : bs

    bs=$4
    if [ -f $2 ]; then
        rm $2
    fi

    if [ "inference" == "$3" ]; then
        echo -e 'Batch Size: ' $bs >$2
        cat $1 | grep Test | tail -n6 | head -n5 |
            awk -v bs=${bs} -F ' ' '{a+=$5}END{printf "Performance Benchmark Time: %.3f sec, Throughput: %.2f FPS\n", a/5, bs*5/a}' >>$2
        cat $1 | tail -n2 | grep "Acc@1" | awk -F ' ' '{printf "Accuracy: acc@1 %.2f\n", $3}' >>$2
    elif [ "training" == "$3" ]; then
        echo -e 'Batch Size: ' $bs >$2
        cat $1 | grep Epoch | tail -n1 | awk -v bs=${bs} -F ' ' '{printf "Performance Benchmark Time: %.3f sec, Throughput: %.2f FPS\nAccuracy: Loss %.4f\n", $4, bs/$4, $10}' >>$2
    else
        echo -e 'Invalid input! Only inference or training are supported.'
        exit 0
    fi
}

if [[ $PRECISION == "bf16" ]]; then
    echo "resnet50 ${PRECISION} inference plain nhwc"
    python -u models/image_recognition/pytorch/resnet50v1_5/inference/gpu/main.py \
        -a resnet50 \
        -b ${BATCH_SIZE} \
        --xpu 0 \
        -e \
        --pretrained \
        --num-iterations ${NUM_ITERATIONS} \
        --bf16 1\
        --channels-last \
        --jit-trace \
        ${DATASET_DIR} 2>&1 | tee ${OUTPUT_DIR}/resnet50-jit_${PRECISION}_inf_plain_nhwc_t0_raw.log
    resnet50_log_analysis ${OUTPUT_DIR}/resnet50-jit_${PRECISION}_inf_plain_nhwc_t0_raw.log ${OUTPUT_DIR}/resnet50-jit_${PRECISION}_inf_plain_nhwc_t0.log inference ${BATCH_SIZE}
else
    echo "The specified precision '${PRECISION}' is unsupported."
    exit 1