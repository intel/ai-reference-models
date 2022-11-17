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

echo 'MODEL_DIR='$MODEL_DIR
echo 'OUTPUT_DIR='$OUTPUT_DIR
echo 'CHECKPOINT_DIR='$CHECKPOINT_DIR
echo 'DATASET_DIR='$DATASET_DIR

if [[ -z $OUTPUT_DIR ]]; then
  echo "The required environment variable OUTPUT_DIR has not been set" >&2
  exit 1
fi

# Create the output directory, if it doesn't already exist
mkdir -p $OUTPUT_DIR

# Create an array of input directories that are expected and then verify that they exist
declare -A input_dirs
input_dirs[CHECKPOINT_DIR]=${CHECKPOINT_DIR}
input_dirs[DATASET_DIR]=${DATASET_DIR}

for i in "${!input_dirs[@]}"; do
  var_name=$i
  dir_path=${input_dirs[$i]}
 
  if [[ -z $dir_path ]]; then
    echo "The required environment variable $var_name is empty" >&2
    exit 1
  fi

  if [[ ! -d $dir_path ]]; then
    echo "The $var_name path '$dir_path' does not exist" >&2
    exit 1
  fi
done

mpi_num_proc_arg=""

if [[ -n $MPI_NUM_PROCESSES ]]; then
  mpi_num_proc_arg="--mpi_num_processes=${MPI_NUM_PROCESSES}"
fi

# If batch size env is not mentioned, then the workload will run with the default batch size.
if [ -z "${BATCH_SIZE}"]; then
  BATCH_SIZE="24"
  echo "Running with default batch size of ${BATCH_SIZE}"
fi

source "${MODEL_DIR}/quickstart/common/utils.sh"
_command python ${MODEL_DIR}/benchmarks/launch_benchmark.py \
--model-name=bert_large \
--precision=fp32 \
--mode=training \
--framework=tensorflow \
--batch-size=${BATCH_SIZE} \
${mpi_num_proc_arg} \
--output-dir $OUTPUT_DIR \
$@ \
-- train_option=SQuAD \
vocab_file=$CHECKPOINT_DIR/vocab.txt \
config_file=$CHECKPOINT_DIR/bert_config.json \
init_checkpoint=$CHECKPOINT_DIR/bert_model.ckpt \
do_train=True \
train_file=$DATASET_DIR/train-v1.1.json \
do_predict=True \
predict_file=$DATASET_DIR/mini-dev-v1.1.json \
learning_rate=3e-5 \
num_train_epochs=0.01 \
max_seq_length=384 \
doc_stride=128 \
output_dir=./large \
optimized_softmax=True \
experimental_gelu=False \
do_lower_case=True

