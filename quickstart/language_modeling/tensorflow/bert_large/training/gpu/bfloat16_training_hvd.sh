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

export MODEL_DIR=${MODEL_DIR-$PWD}

echo 'MODEL_DIR='$MODEL_DIR
echo 'OUTPUT_DIR='$OUTPUT_DIR
echo 'BERT_LARGE_DIR='$BERT_LARGE_DIR

if [[ -z $OUTPUT_DIR ]]; then
  echo "The required environment variable OUTPUT_DIR has not been set" >&2
  exit 1
fi

# Create the output directory, if it doesn't already exist
mkdir -p $OUTPUT_DIR

# Create an array of input directories that are expected and then verify that they exist
declare -A input_dirs
input_dirs[BERT_LARGE_DIR]=${BERT_LARGE_DIR}

for i in "${!input_dirs[@]}"; do
  var_name=$i
  dir_path=${input_dirs[$i]}
 
  if [[ -z $dir_path ]]; then
    echo "The required environment variable $var_name is empty" >&2
    exit 1
  fi
done

# Check for precision
if [[ $PRECISION != "bfloat16" ]]; then
  echo "The specified precision '${PRECISION}' is unsupported."
  echo "Only bfloat16 precision is supported"
  exit 1
fi

# Create Dummy dataset from bert large dataset
export DUMMY_DATA=${OUTPUT_DIR}/tf-examples-512.tfrecord
echo 'DUMMY_DATA='$DUMMY_DATA
$MODEL_DIR/quickstart/language_modeling/tensorflow/bert_large/training/gpu/generate_pretraining_data.sh

export TF_NUM_INTEROP_THREADS=1

export NUMBER_OF_PROCESS=2
export PROCESS_PER_NODE=2

if [ -z "${BATCH_SIZE}" ]; then
      BATCH_SIZE="32"
      echo "Running with default batch size of ${BATCH_SIZE}"
fi

source "${MODEL_DIR}/quickstart/common/utils.sh"
_command mpirun -np $NUMBER_OF_PROCESS -ppn $PROCESS_PER_NODE --prepend-rank \
  python ${MODEL_DIR}/models/language_modeling/tensorflow/bert_large/training/bfloat16/run_pretraining.py \
  --input_file=$DUMMY_DATA \
  --output_dir=${OUTPUT_DIR} \
  --precision=${PRECISION} \
  --do_train=True \
  --do_eval=False \
  --bert_config_file=${BERT_LARGE_DIR}/bert_config.json \
  --train_batch_size=${BATCH_SIZE} \
  --max_seq_length=512 \
  --max_predictions_per_seq=76 \
  --num_train_steps=120 \
  --num_warmup_steps=6 \
  --accum_steps=1 \
  --learning_rate=2e-5 \
  --do_lower_case=False \
  --mpi_workers_sync_gradients=False \
  --use_tpu=False \
  --experimental_gelu=True \
  --optimized_softmax=True \
  --inter_op_parallelism_threads=1 \
  --intra_op_parallelism_threads=1 2>&1 | tee ${OUTPUT_DIR}//bfloat16_trn_hvd.log
