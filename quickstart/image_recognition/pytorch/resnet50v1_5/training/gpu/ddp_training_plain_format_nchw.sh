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

# If batch size env is not mentioned, then the workload will run with the default batch size.
if [ -z "${BATCH_SIZE}"]; then
  BATCH_SIZE="256"
  echo "Running with default batch size of ${BATCH_SIZE}"
fi

# Create the output directory, if it doesn't already exist
mkdir -p $OUTPUT_DIR

export LD_PRELOAD=/opt/intel/oneapi/lib/intel64/libmpi.so

echo "explicit scaling hvd_resnet50 bf16 training plain nhwc bs16 perf 1c2t"
cd ${MODEL_DIR}/models/image_recognition/pytorch/resnet50v1_5/training/gpu
I_MPI_DEBUG=6 mpiexec -np 2 -ppn 2 python main.py -a resnet50 \
    -b ${BATCH_SIZE} \
    --xpu 0 \
    ${DATASET_DIR} \
    --num-iterations 20 \
    --bucket-cap 200 \
    --broadcast-buffers False \
    --bf16 1 2>&1 | tee ${OUTPUT_DIR}/ddp-resnet50_bf16_train_block_nchw_1c2t_raw.log
cp ${OUTPUT_DIR}/ddp-resnet50_bf16_train_block_nchw_1c2t_raw.log ${OUTPUT_DIR}/ddp-resnet50_bf16_train_block_nchw_1c2t.log
cd -
