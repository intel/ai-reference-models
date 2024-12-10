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

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

if [ -z "${DATASET_DIR}" ]; then
  echo "The required environment variable DATASET_DIR has not been set"
  exit 1
fi

if [ ! -d "${DATASET_DIR}" ]; then
  echo "The DATASET_DIR '${DATASET_DIR}' does not exist"
  exit 1
fi

if [ -z "${PRECISION}" ]; then
  echo "The required environment variable PRECISION has not been set"
  echo "Please set PRECISION to fp32, bfloat16 or fp16."
  exit 1
elif [ ${PRECISION} != "fp32" ] && [ ${PRECISION} != "bfloat16" ] && [ ${PRECISION} != "fp16" ]; then
  echo "The specified precision '${PRECISION}' is unsupported."
  echo "Supported precisions are: fp32, bfloat16 and fp16"
  exit 1
fi

if [ -z "${BATCH_SIZE}" ]; then
  BATCH_SIZE="24"
  echo "Running with default batch size of ${BATCH_SIZE}"
fi

cores_per_socket=$(lscpu |grep 'Core(s) per socket:' |sed 's/[^0-9]//g')
cores_per_socket="${cores_per_socket//[[:blank:]]/}"

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
NUMAS=`lscpu | grep 'NUMA node(s)' | awk '{print $3}'`
CORES_PER_NUMA=`expr $CORES \* $SOCKETS / $NUMAS`

NUM_INSTANCES=`expr $cores_per_socket / $CORES_PER_NUMA`
export OMP_NUM_THREADS=${cores_per_socket}

source "${MODEL_DIR}/models_v2/common/utils.sh"
_ht_status_spr
_command python ${MODEL_DIR}/benchmarks/launch_benchmark.py \
  --model-name=bert_large \
  --precision=${PRECISION} \
  --mode=training \
  --framework tensorflow \
  --output-dir ${OUTPUT_DIR} \
  --mpi_num_processes=${NUM_INSTANCES} \
  --mpi_num_processes_per_socket=${NUM_INSTANCES} \
  --batch-size ${BATCH_SIZE} \
  --num-intra-threads $CORES_PER_NUMA \
  --num-inter-threads 2 \
  $@ \
  -- DEBIAN_FRONTEND=noninteractive \
  train-option=SQuAD do-predict=False do-train=True profile=False \
  learning-rate=3e-5 max-seq-length=384 \
  save-checkpoints_steps=1000 \
  config_file=${DATASET_DIR}/wwm_uncased_L-24_H-1024_A-16/bert_config.json \
  init_checkpoint=${DATASET_DIR}/wwm_uncased_L-24_H-1024_A-16/bert_model.ckpt \
  vocab_file=${DATASET_DIR}/wwm_uncased_L-24_H-1024_A-16/vocab.txt \
  train_file=${DATASET_DIR}/wwm_uncased_L-24_H-1024_A-16/train-v1.1.json \
  optimized_softmax=True doc_stride=128 num_train_epochs=2 \
  experimental_gelu=False do_lower_case=True 2>&1 | tee ${OUTPUT_DIR}/bert_large_squad_${PRECISION}_training_bs${BATCH_SIZE}_all_instances.log

if [[ $? == 0 ]]; then
  cat ${OUTPUT_DIR}/bert_large_squad_${PRECISION}_training_bs${BATCH_SIZE}_all_instances.log | grep "INFO:tensorflow:examples/sec:" | tail -n 2 | sed -e "s/.*: //"
  exit 0
else
  exit 1
fi
