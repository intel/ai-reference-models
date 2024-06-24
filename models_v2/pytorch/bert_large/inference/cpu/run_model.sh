#!/bin/bash

#
# Copyright (c) 2024 Intel Corporation
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

if [[ "$TEST_MODE" == "THROUGHPUT" ]]; then  
    echo "TEST_MODE set to THROUGHPUT"
elif [[ "$TEST_MODE" == "ACCURACY" ]]; then 
    echo "TEST_MODE set to ACCURACY"
else
    echo "Please set TEST_MODE to THROUGHPUT or ACCURACY"
    exit
fi

ARGS=${ARGS:-""}

#export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
precision=fp32

if [[ "$PRECISION" == *"avx"* ]]; then
    unset DNNL_MAX_CPU_ISA
fi

if [[ "$PRECISION" == "bf16" ]]
then
    precision=bf16
    ARGS="$ARGS --bf16"
    echo "### running bf16 mode"
elif [[ "$PRECISION" == "fp16" ]]
then
    precision=fp16
    ARGS="$ARGS --fp16_cpu"
    echo "### running fp16 mode"

elif [[ "$PRECISION" == "bf32" ]]
then
    precision=bf32
    ARGS="$ARGS --bf32"
    echo "### running bf32 mode"
elif [[ "$PRECISION" == "int8" || "$PRECISION" == "avx-int8" ]]
then
    precision=int8
    ARGS="$ARGS --int8 --int8_bf16"
    echo "### running int8 mode"
elif [[ "$PRECISION" == "fp32" || "$PRECISION" == "avx-fp32" ]]
then
    precision=fp32
    echo "### running fp32 mode"
else
    echo "Please set PRECISION to : fp32, int8, bf32, bf26, avx-int8 or avx-fp32"
    exit
fi

export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000";
BATCH_SIZE=${BATCH_SIZE:-56}
EVAL_DATA_FILE=${EVAL_DATA_FILE:-"${PWD}/squad1.1/dev-v1.1.json"}
FINETUNED_MODEL=${FINETUNED_MODEL:-"bert_squad_model"}
OUTPUT_DIR=${OUTPUT_DIR:-${PWD}}
EVAL_SCRIPT=${EVAL_SCRIPT:-"./transformers/examples/legacy/question-answering/run_squad.py"}
work_space=${work_space:-${OUTPUT_DIR}}
INT8_CONFIG=${INT8_CONFIG:-"configure.json"}


rm -rf ${OUTPUT_DIR}/throughput_log*

TORCH_INDUCTOR=${TORCH_INDUCTOR:-"0"}

if [ "$WEIGHT_SHARING" ]; then
    CORES=`lscpu | grep Core | awk '{print $4}'`
    SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
    TOTAL_CORES=`expr $CORES \* $SOCKETS`
    CORES_PER_INSTANCE=$CORES
    INSTANCES=`expr $TOTAL_CORES / $CORES_PER_INSTANCE`
    LAST_INSTANCE=`expr $INSTANCES - 1`
    INSTANCES_PER_SOCKET=`expr $INSTANCES / $SOCKETS`
    STREAM_PER_INSTANCE=$CORES_PER_INSTANCE
    BATCH_SIZE=$STREAM_PER_INSTANCE

    if [[ "$TEST_MODE" == "THROUGHPUT" ]]; then
        echo "Running Bert_Large inference throughput with runtime extension enabled."
        for i in $(seq 0 $LAST_INSTANCE); do
            numa_node_i=`expr $i / $INSTANCES_PER_SOCKET`
            start_core_i=`expr $i \* $CORES_PER_INSTANCE`
            end_core_i=`expr $start_core_i + $CORES_PER_INSTANCE - 1`
            LOG_i="${OUTPUT_DIR}/throughput_log_${PRECISION}_${i}.log"

            ARGS="$ARGS --use_multi_stream_module"
            ARGS="$ARGS --num_streams $STREAM_PER_INSTANCE"
            ARGS="$ARGS --instance_number $numa_node_i"

            numactl -C $start_core_i-$end_core_i --membind=$numa_node_i python ${EVAL_SCRIPT} $ARGS --model_type bert --model_name_or_path ${FINETUNED_MODEL} --tokenizer_name bert-large-uncased-whole-word-masking-finetuned-squad  --do_eval --do_lower_case --predict_file $EVAL_DATA_FILE --per_gpu_eval_batch_size $BATCH_SIZE --learning_rate 3e-5 --num_train_epochs 2.0 --max_seq_length 384 --doc_stride 128 --output_dir ./tmp --perf_begin_iter 15 --use_jit --ipex --perf_run_iters 40 \
            2>&1 | tee ${LOG_i} &
        done
        wait
    else
        echo "Running Bert_Large inference accuracy with runtime extension enabled."
        numa_node_i=0
        start_core_i=0
        end_core_i=`expr $start_core_i + $CORES_PER_INSTANCE - 1`
        LOG_0="${OUTPUT_DIR}/accuracy_log_${PRECISION}.log"
        
        ARGS="$ARGS --use_multi_stream_module"
        ARGS="$ARGS --num_streams $STREAM_PER_INSTANCE"
        ARGS="$ARGS --instance_number $numa_node_i"

        numactl -C $start_core_i-$end_core_i --membind=$numa_node_i python $EVAL_SCRIPT $ARGS --model_type bert --model_name_or_path ${FINETUNED_MODEL}  --do_eval --do_lower_case --predict_file $EVAL_DATA_FILE  --per_gpu_eval_batch_size $BATCH_SIZE --learning_rate 3e-5 --num_train_epochs 2.0 --max_seq_length 384 --doc_stride 128 --output_dir ./tmp --tokenizer_name bert-large-uncased-whole-word-masking-finetuned-squad --use_jit --ipex --int8_config ${INT8_CONFIG} \
        2>&1 | tee $LOG_0
    fi
elif [[ "0" == "$TORCH_INDUCTOR" ]]; then
    if [[ "$TEST_MODE" == "THROUGHPUT" ]]; then 
        python -m intel_extension_for_pytorch.cpu.launch --throughput_mode --enable_jemalloc --log_path=${OUTPUT_DIR} --log_file_prefix="./throughput_log_${precision}" ${EVAL_SCRIPT} $ARGS --model_type bert --model_name_or_path ${FINETUNED_MODEL} --tokenizer_name bert-large-uncased-whole-word-masking-finetuned-squad  --do_eval --do_lower_case --predict_file $EVAL_DATA_FILE --per_gpu_eval_batch_size $BATCH_SIZE --learning_rate 3e-5 --num_train_epochs 2.0 --max_seq_length 384 --doc_stride 128 --output_dir ./tmp
    else 
        if [[ "$PRECISION" == "fp8" ]];then
                python -m intel_extension_for_pytorch.cpu.launch --log_path=${OUTPUT_DIR} --log_file_prefix="accuracy_log" $EVAL_SCRIPT $ARGS --model_type bert --model_name_or_path ${FINETUNED_MODEL}  --do_eval --do_lower_case --predict_file $EVAL_DATA_FILE  --per_gpu_eval_batch_size $BATCH_SIZE --learning_rate 3e-5 --num_train_epochs 2.0 --max_seq_length 384 --doc_stride 128 --output_dir ./tmp --tokenizer_name bert-large-uncased-whole-word-masking-finetuned-squad --ipex --fp8_config ${FP8_CONFIG} 2>&1
        else
            python -m intel_extension_for_pytorch.cpu.launch --log_path=${OUTPUT_DIR} --log_file_prefix="accuracy_log" $EVAL_SCRIPT $ARGS --model_type bert --model_name_or_path ${FINETUNED_MODEL}  --do_eval --do_lower_case --predict_file $EVAL_DATA_FILE  --per_gpu_eval_batch_size $BATCH_SIZE --learning_rate 3e-5 --num_train_epochs 2.0 --max_seq_length 384 --doc_stride 128 --output_dir ./tmp --tokenizer_name bert-large-uncased-whole-word-masking-finetuned-squad --use_jit --ipex --int8_config ${INT8_CONFIG} 2>&1
        fi
    fi 
else
    echo "Running Bert_Large inference with torch.compile() indutor backend enabled."
    export TORCHINDUCTOR_FREEZING=1
    if [[ "$TEST_MODE" == "THROUGHPUT" ]]; then 
        python -m intel_extension_for_pytorch.cpu.launch --throughput_mode --enable_jemalloc --log_path=${OUTPUT_DIR} --log_file_prefix="./throughput_log_${precision}" ${EVAL_SCRIPT} $ARGS --model_type bert --model_name_or_path ${FINETUNED_MODEL} --tokenizer_name bert-large-uncased-whole-word-masking-finetuned-squad  --do_eval --do_lower_case --predict_file $EVAL_DATA_FILE --per_gpu_eval_batch_size $BATCH_SIZE --learning_rate 3e-5 --num_train_epochs 2.0 --max_seq_length 384 --doc_stride 128 --output_dir ./tmp --perf_begin_iter 15 --inductor --perf_run_iters 40 --int8_config ${INT8_CONFIG}
    else 
        python -m intel_extension_for_pytorch.cpu.launch --log_path=${OUTPUT_DIR} --log_file_prefix="accuracy_log" $EVAL_SCRIPT $ARGS --model_type bert --model_name_or_path ${FINETUNED_MODEL}  --do_eval --do_lower_case --predict_file $EVAL_DATA_FILE  --per_gpu_eval_batch_size $BATCH_SIZE --learning_rate 3e-5 --num_train_epochs 2.0 --max_seq_length 384 --doc_stride 128 --output_dir ./tmp --tokenizer_name bert-large-uncased-whole-word-masking-finetuned-squad --inductor --int8_config ${INT8_CONFIG} 2>&1
    fi
fi 

if [[ "$TEST_MODE" == "THROUGHPUT" ]]; then 
    logs=throughput_log* 
else 
    logs=accuracy_log* 
fi

total_throughput=0
total_latency=0 
total_accuracy=0
num_logs=0

for log_file in ${OUTPUT_DIR}/${logs}; do
    throughput=$(grep -oP "'total': \K\d+" $log_file)
    if [ -z "$throughput" ]; then
        continue  
    fi 
    latency=$(grep -oP "\(.* sec per example\)" ${log_file} | grep -oP "\d+\.\d+")
    evaluation_time=$(grep -oP "Evaluation done in total \K\d+\.\d+" $log_file)
    accuracy=$(grep -oP "'f1': \K\d+\.\d+" ${log_file})
    if [ -z "$evaluation_time" ]; then
        continue
    fi
    
    average_time_per_example=$(bc <<< "scale=3; $evaluation_time / $throughput")
    throughput=$(bc <<< "scale=2; 1 / $average_time_per_example")
    
    total_throughput=$(bc <<< "$total_throughput + $throughput")
    total_latency=$(bc <<< "$total_latency + $latency")
    total_accuracy=$(bc <<< "$total_accuracy + $accuracy")
    ((num_logs++))
done

if [ $num_logs -gt 0 ]; then
    average_throughput=$(bc <<< "scale=3; $total_throughput / $num_logs")
    average_latency=$(bc <<< "scale=3; $total_latency / $num_logs")
    average_accuracy=$(bc <<< "scale=3; $total_accuracy / $num_logs")

    echo "Average throughput across all valid logs: $average_throughput examples per second" | tee -a ${OUTPUT_DIR}/summary.log
    echo "Average latency across all valid logs: $average_latency seconds per example" | tee -a ${OUTPUT_DIR}/summary.log
    echo "Average accuracy across all valid logs: $average_accuracy %" | tee -a ${OUTPUT_DIR}/summary.log

else
    echo "No valid throughput/accuracy logs found for calculation." | tee -a ${OUTPUT_DIR}/summary.log
    exit
fi

yaml_content=$(cat << EOF
results: 
- key : throughput
  value: $average_throughput
  unit: examples per second 
- key: latency
  value: $average_latency
  unit: seconds per example
- key: accuracy
  value: $average_accuracy
  unit: percentage
EOF
)

echo "$yaml_content" >  $OUTPUT_DIR/results.yaml
echo "YAML file created."
