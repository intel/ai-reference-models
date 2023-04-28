#!/usr/bin/env bash
#
# Copyright (c) 2022-2023 Intel Corporation
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


BATCH_SIZE=${BATCH_SIZE-16}


if [[ -z "${Tile}" ]]; then
    Tile=${Tile-1}
else
    Tile=${Tile}
fi

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "The DATASET_DIR '${DATASET_DIR}' does not exist"
  exit 1
fi

if [[ -z $OUTPUT_DIR ]]; then
  echo "The required environment variable OUTPUT_DIR has not been set" >&2
  exit 1
fi

# Create the output directory, if it doesn't already exist
mkdir -p $OUTPUT_DIR


bert_log_analysis() {
    # $1 : src raw log
    # $2 : dst format log
    # $3 : inference or training
    # $4 : bs

    if [ -f $2 ]; then
        rm $2
    fi

    bs=$4
    if [ "training" == "$3" ]; then
        echo -e 'Batch Size: ' $bs >$2
        grep "train perf: " $1 | tail -n1 | awk -v bs=${bs} -F ' ' '{printf "Performance Benchmark Time: %.3f sec, Throughput: %.2f seq/sec\n", $3, bs/$3}' >>$2
        grep "perplexity = " $1 | awk -F ' ' '{printf "Accuracy: perplexity %.6f\n", $NF}' >>$2
    else
        echo -e 'Invalid input! Only training are supported.'
        exit 0
    fi
}

if [[ ! -d ${PROCESSED_DATASET_DIR}/hdf5_seq_512 ]]; then
  if [[ ! -d ${DATASET_DIR}/results4 ]]; then
    gdown https://drive.google.com/uc?id=14xV2OUGSQDG_yDBrmbSdcDC-QGeqpfs_
    tar -xf results_text.tar.gz
    chmod 775 results4
    mv results4 ${DATASET_DIR}
  fi
  cd ${MODEL_DIR}/models/language_modeling/pytorch/bert_large/training/gpu/data/
  bash parallel_create_hdf5.sh
  cd -
fi


if [[ ${Tile} == "1" ]]; then
  echo "bert bf16 training plain nchw"
  cd ${MODEL_DIR}/models/language_modeling/pytorch/bert_large/training/gpu/
  python run_pretrain_mlperf.py \
      --config_name=bert_config.json \
      --input_dir=${PROCESSED_DATASET_DIR}/hdf5_seq_512 \
      --output_dir=result \
      --eval_dir=${PROCESSED_DATASET_DIR}/hdf5_seq_512 \
      --device=xpu \
      --do_train \
      --train_batch_size=${BATCH_SIZE} \
      --gradient_accumulation_steps=1 \
      --bf16 \
      --adamw --num-iterations 10 2>&1 | tee ${OUTPUT_DIR}/bert_bf16_train_plain_nchw_t0_raw.log
  wait
  bert_log_analysis ${OUTPUT_DIR}/bert_bf16_train_plain_nchw_t0_raw.log ${OUTPUT_DIR}/bert_bf16_train_plain_nchw_t0.log training ${BATCH_SIZE}
  cd -
elif [[ ${Tile} == "2" ]]; then
  echo "bert bf16 training plain nchw 2 tile"
  cd ${MODEL_DIR}/models/language_modeling/pytorch/bert_large/training/gpu/
  ZE_AFFINITY_MASK=0.0 python run_pretrain_mlperf.py \
      --config_name=bert_config.json \
      --input_dir=${PROCESSED_DATASET_DIR}/hdf5_seq_512 \
      --output_dir=result \
      --eval_dir=${PROCESSED_DATASET_DIR}/hdf5_seq_512 \
      --device=xpu \
      --do_train \
      --train_batch_size=${BATCH_SIZE}  \
      --gradient_accumulation_steps=1 \
      --bf16 \
      --adamw --num-iterations 10 2>&1 | tee ${OUTPUT_DIR}/bert_bf16_train_plain_nchw_t0_raw.log &
  ZE_AFFINITY_MASK=0.1 python run_pretrain_mlperf.py \
      --config_name=bert_config.json \
      --input_dir=${PROCESSED_DATASET_DIR}/hdf5_seq_512\
      --output_dir=result \
      --eval_dir=${PROCESSED_DATASET_DIR}/hdf5_seq_512 \
      --device=xpu \
      --do_train \
      --train_batch_size=${BATCH_SIZE}  \
      --gradient_accumulation_steps=1 \
      --bf16 \
      --adamw --num-iterations 10  2>&1 | tee ${OUTPUT_DIR}/bert_bf16_train_plain_nchw_t1_raw.log
  wait
  bert_log_analysis ${OUTPUT_DIR}/bert_bf16_train_plain_nchw_t0_raw.log ${OUTPUT_DIR}/bert_bf16_train_plain_nchw_t0.log training ${BATCH_SIZE}
  bert_log_analysis ${OUTPUT_DIR}/bert_bf16_train_plain_nchw_t1_raw.log ${OUTPUT_DIR}/bert_bf16_train_plain_nchw_t1.log training ${BATCH_SIZE}
  cd -
else
    echo "The specified Tile '${Tile}' is unsupported."
    echo "Supported tile number are: 1 and 2"
    exit 1
fi
