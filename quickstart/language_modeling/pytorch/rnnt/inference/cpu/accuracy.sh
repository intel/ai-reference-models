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

if [ ! -e "${MODEL_DIR}/models/language_modeling/pytorch/rnnt/inference/cpu/inference.py" ]; then
  echo "Could not find the script of inference.py. Please set environment variable '\${MODEL_DIR}'."
  echo "From which the inference.py exist at the: \${MODEL_DIR}/models/language_modeling/pytorch/rnnt/inference/cpu/inference.py"
  exit 1
fi

if [ ! -e "${CHECKPOINT_DIR}/results/rnnt.pt" ]; then
  echo "The pretrained model \${CHECKPOINT_DIR}/results/rnnt.pt does not exist"
  exit 1
fi

if [ ! -d "${DATASET_DIR}/dataset/LibriSpeech" ]; then
  echo "The DATASET_DIR \${DATASET_DIR}/dataset/LibriSpeech does not exist"
  exit 1
fi

if [ ! -d "${OUTPUT_DIR}" ]; then
  echo "The OUTPUT_DIR '${OUTPUT_DIR}' does not exist"
  exit 1
fi

if [[ "$1" == *"avx"* ]]; then
    unset DNNL_MAX_CPU_ISA
fi

ARGS=""
if [ "$1" == "bf16" ]; then
    ARGS="$ARGS --mix-precision"
    echo "### running bf16 datatype"
else
    echo "### running fp32 datatype"
fi

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export KMP_BLOCKTIME=1
export KMP_AFFINITY=granularity=fine,compact,1,0

BATCH_SIZE=64
PRECISION=$1

rm -rf ${OUTPUT_DIR}/rnnt_${PRECISION}_inference_accuracy*

python -m intel_extension_for_pytorch.cpu.launch \
    --use_default_allocator \
    ${MODEL_DIR}/models/language_modeling/pytorch/rnnt/inference/cpu/inference.py \
    --dataset_dir ${DATASET_DIR}/dataset/LibriSpeech/ \
    --val_manifest ${DATASET_DIR}/dataset/LibriSpeech/librispeech-dev-clean-wav.json \
    --model_toml ${MODEL_DIR}/models/language_modeling/pytorch/rnnt/inference/cpu/configs/rnnt.toml \
    --ckpt ${CHECKPOINT_DIR}/results/rnnt.pt \
    --batch_size $BATCH_SIZE \
    --ipex \
    --jit \
    $ARGS 2>&1 | tee ${OUTPUT_DIR}/rnnt_${PRECISION}_inference_accuracy.log

# For the summary of results
wait

accuracy=$(grep 'Accuracy:' ${OUTPUT_DIR}/rnnt_${PRECISION}_inference_accuracy* |sed -e 's/.*Accuracy//;s/[^0-9.]//g')
WER=$(grep 'Evaluation WER:' ${OUTPUT_DIR}/rnnt_${PRECISION}_inference_accuracy* |sed -e 's/.*Evaluation WER//;s/[^0-9.]//g')
echo ""RNN-T";"accuracy";$1; ${BATCH_SIZE};${accuracy}" | tee -a ${OUTPUT_DIR}/summary.log
echo ""RNN-T";"WER";$1; ${BATCH_SIZE};${WER}" | tee -a ${work_space}/summary.log
