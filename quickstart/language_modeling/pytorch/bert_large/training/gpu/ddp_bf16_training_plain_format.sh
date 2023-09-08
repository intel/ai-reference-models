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

source $(python -c "import oneccl_bindings_for_pytorch as torch_ccl;print(torch_ccl.cwd)")/env/setvars.sh
export LD_PRELOAD=$(python -c "import oneccl_bindings_for_pytorch as torch_ccl;print(torch_ccl.cwd)")/lib/libmpi.so
export ONECCL_BINDINGS_FOR_PYTORCH_ENV_VERBOSE=1


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

echo "explicit scaling ddp bert bf16 training plain nchw 1c2t"
cd ${MODEL_DIR}/models/language_modeling/pytorch/bert_large/training/gpu/
I_MPI_DEBUG=6 mpiexec -np 2 -ppn 2 python run_pretrain_mlperf.py \
    --config_name=bert_config.json \
    --input_dir=${PROCESSED_DATASET_DIR}/hdf5_seq_512 \
    --output_dir=result \
    --eval_dir=${PROCESSED_DATASET_DIR}/hdf5_seq_512 \
    --device=xpu \
    --do_train \
    --train_batch_size=${BATCH_SIZE} \
    --gradient_accumulation_steps=1 \
    --bf16 \
    --adamw --num-iterations 10 2>&1 | tee ${OUTPUT_DIR}/ddp-bert_bf16_train_plain_nchw_1c2t_raw.log
wait
bert_log_analysis ${OUTPUT_DIR}/ddp-bert_bf16_train_plain_nchw_1c2t_raw.log ${OUTPUT_DIR}/ddp-bert_bf16_train_plain_nchw_1c2t.log training ${BATCH_SIZE}
cd -
