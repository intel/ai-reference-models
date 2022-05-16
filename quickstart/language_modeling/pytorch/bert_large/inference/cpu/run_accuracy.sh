#!/bin/bash

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

#export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
ARGS=""
precision=fp32

if [[ "$1" == *"avx"* ]]; then
    unset DNNL_MAX_CPU_ISA
fi

if [[ "$1" == "bf16" ]]
then
    precision=bf16
    ARGS="$ARGS --bf16"
    echo "### running bf16 mode"
elif [[ "$1" == "int8" || "$1" == "avx-int8" ]]
then
    precision=int8
    ARGS="$ARGS --int8"
    echo "### running int8 mode"
elif [[ "$1" == "fp32" || "$1" == "avx-fp32" ]]
then
    precision=fp32
    echo "### running fp32 mode"
fi

rm -f ${OUTPUT_DIR}/accuracy_log*
INT8_CONFIG=${INT8_CONFIG:-"configure.json"}
BATCH_SIZE=${BATCH_SIZE:-8}
EVAL_DATA_FILE=${EVAL_DATA_FILE:-"${PWD}/squad1.1/dev-v1.1.json"}
FINETUNED_MODEL=${FINETUNED_MODEL:-bert_squad_model}
OUTPUT_DIR=${OUTPUT_DIR:-"${PWD}"}
EVAL_SCRIPT=${EVAL_SCRIPT:-"./transformers/examples/question-answering/run_squad.py"}
work_space=${work_space:-"${OUTPUT_DIR}"}
python -m intel_extension_for_pytorch.cpu.launch --log_path=${OUTPUT_DIR} --log_file_prefix="accuracy_log" $EVAL_SCRIPT $ARGS --model_type bert --model_name_or_path ${FINETUNED_MODEL}  --do_eval --do_lower_case --predict_file $EVAL_DATA_FILE  --per_gpu_eval_batch_size $BATCH_SIZE --learning_rate 3e-5 --num_train_epochs 2.0 --max_seq_length 384 --doc_stride 128 --output_dir ./tmp --tokenizer_name bert-large-uncased-whole-word-masking-finetuned-squad --use_jit --int8_config ${INT8_CONFIG} 2>&1 | tee $LOG_0

accuracy=$(grep 'Results:' ${OUTPUT_DIR}/accuracy_log*|awk -F ' ' '{print $12}' | awk -F ',' '{print $1}')
echo ""BERT";"f1";${precision}; ${BATCH_SIZE};${accuracy}" | tee -a ${OUTPUT_DIR}/summary.log


