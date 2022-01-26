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
ARGS="--benchmark"
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

export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000";
INT8_CONFIG=${INT8_CONFIG:-"configure.json"}
BATCH_SIZE=${BATCH_SIZE:-56}
EVAL_DATA_FILE=${EVAL_DATA_FILE:-"${PWD}/squad1.1/dev-v1.1.json"}
FINETUNED_MODEL=${FINETUNED_MODEL:-"bert_squad_model"}
OUTPUT_DIR=${OUTPUT_DIR:-${PWD}}
EVAL_SCRIPT=${EVAL_SCRIPT:-"./transformers/examples/question-answering/run_squad.py"}
work_space=${work_space:-${OUTPUT_DIR}}

rm -rf ${OUTPUT_DIR}/throughput_log*
python -m intel_extension_for_pytorch.cpu.launch --throughput_mode --enable_jemalloc --log_path=${OUTPUT_DIR} --log_file_prefix="./throughput_log_${precision}" ${EVAL_SCRIPT} $ARGS --model_type bert --model_name_or_path ${FINETUNED_MODEL} --tokenizer_name bert-large-uncased-whole-word-masking-finetuned-squad  --do_eval --do_lower_case --predict_file $EVAL_DATA_FILE --per_gpu_eval_batch_size $BATCH_SIZE --learning_rate 3e-5 --num_train_epochs 2.0 --max_seq_length 384 --doc_stride 128 --output_dir ./tmp --perf_begin_iter 15 --use_jit --perf_run_iters 40 --int8_config ${INT8_CONFIG}

throughput=$(grep 'Throughput:' ${OUTPUT_DIR}/throughput_log* |sed -e 's/.*Throughput//;s/[^0-9.]//g' |awk '
BEGIN {
        sum = 0;
	i = 0;
      }
      {
        sum = sum + $1;
i++;
      }
END   {
sum = sum / i;
printf("%.3f", sum);
}')
echo ""BERT";"throughput";${precision}; ${BATCH_SIZE};${throughput}" | tee -a ${OUTPUT_DIR}/summary.log
