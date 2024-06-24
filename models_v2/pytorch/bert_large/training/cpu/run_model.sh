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

if [[ "$TRAINING_PHASE" == '1' ]]; then 
    echo "Running phase 1 training" 
elif [ "$TRAINING_PHASE" == '2' ]; then
    echo "Running phase 2 training"
else
    echo "Please set TRAINING_PHASE to 1 or 2"
    exit 1
fi 

if [[ "$DDP" == 'true' ]]; then 
    echo "Running distributed training" 
    oneccl_bindings_for_pytorch_path=$(python -c "import torch; import oneccl_bindings_for_pytorch; import os;  print(os.path.abspath(os.path.dirname(oneccl_bindings_for_pytorch.__file__)))")
    source $oneccl_bindings_for_pytorch_path/env/setvars.sh
elif [ "$DDP" == 'false' ]; then
    echo "Running single-node training"
else
    echo "Please set DDP to true or false"
    exit 1
fi 

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

if [ -z "${PRECISION}" ]; then
  echo "The required environment variable PRECISION has not been set"
  exit 1
fi

if [ -z "${DATASET_DIR}" ]; then
  echo "The required environment variable DATASET has not been set"
  exit 1
fi

export TRAIN_SCRIPT=${PWD}/run_pretrain_mlperf.py

MODEL_DIR=${MODEL_DIR-$PWD}

#export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
ARGS="$ARGS --benchmark"
precision=fp32

batch_size=${batch_size:-224}
if [[ "$PRECISION" == *"avx"* ]]; then
    unset DNNL_MAX_CPU_ISA
fi

if [[ "$PRECISION" == "bf16" ]]; then
    ARGS="$ARGS --bf16"
    precision=bf16
    batch_size=${batch_size:-448}
    echo "### running bf16 mode"
elif [[ $PRECISION == "bf32" ]]; then
    echo "### running BF32 mode"
    ARGS="$ARGS --bf32"
    precision=bf32
elif [[ $DDP == 'false' && $PRECISION == "fp16" ]]; then
    echo "### running FP16 mode"
    ARGS="$ARGS --fp16"
    precision=fp16
elif [[ $DDP == 'false' && $PRECISION == "fp8" ]]; then
    echo "### running FP8 mode"
    ARGS="$ARGS --fp8"
    precision=fp8
elif [[ $PRECISION == "fp32" || $PRECISION == "avx-fp32" ]]; then
    echo "### running FP32 mode"

else
    echo "The specified precision '$PRECISION' is unsupported."
    echo "Supported precisions for single-node training are: fp32, bf32, avx-fp32, bf16, fp8"
    echo "Supported precisions for distributed training are: fp32, bf16, bf32"
    exit 1
fi


DATASET_DIR=${DATASET_DIR:-~/dataset/}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-${MODEL_DIR}/run_pretrain_mlperf.py}
OUTPUT_DIR=${OUTPUT_DIR:-${PWD}}
work_space=${work_space:-${OUTPUT_DIR}}

ARGS_IPEX=""
params=""


if [[ "$DDP" == "true" ]]; then 
    SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
    NNODES=${NNODES:-1}
    NUM_RANKS=$(( NNODES * SOCKETS ))
    LBS=$(( batch_size / NUM_RANKS ))
    export FI_PROVIDER=psm3
    export PSM3_HAL=sockets
    params="$params --log_freq=0" 

    ARGS_IPEX="$ARGS_IPEX --distributed"
    ARGS_IPEX="$ARGS_IPEX --nnodes ${NNODES}"
    ARGS_IPEX="$ARGS_IPEX --hostfile ${HOSTFILE}"
    ARGS_IPEX="$ARGS_IPEX --nproc_per_node ${SOCKETS}"
else 
    NUM_RANKS=1
    LBS=$(( batch_size / NUM_RANKS ))
    params="$params --num_samples_per_checkpoint 1 --min_samples_to_start_checkpoints 1 --log_freq 1"
    
    ARGS_IPEX="$ARGS_IPEX --node_id 0"
    ARGS_IPEX="$ARGS_IPEX --enable_jemalloc"
fi 

params="$params --train_batch_size=$LBS     --learning_rate=3.5e-4     --opt_lamb_beta_1=0.9     --opt_lamb_beta_2=0.999     --warmup_proportion=0.0     --warmup_steps=0.0     --start_warmup_step=0     --max_steps=13700   --max_predictions_per_seq=76      --do_train     --skip_checkpoint     --train_mlm_accuracy_window_size=0     --target_mlm_accuracy=0.720     --weight_decay_rate=0.01     --max_samples_termination=4500000     --eval_iter_start_samples=150000 --eval_iter_samples=150000     --eval_batch_size=16  --gradient_accumulation_steps=1"

if [[ "$TRAINING_PHASE" == "1" ]]; then
    rm -rf ${OUTPUT_DIR}/throughput_log_phase1_*
    LOG_PREFIX="throughput_log_phase1_${precision}"

    BERT_MODEL_CONFIG=${BERT_MODEL_CONFIG-~/dataset/checkpoint/config.json}
    CHECKPOINT_DIR=${CHECKPOINT_DIR-${PWD}/checkpoint_phase1_dir}

    TORCH_INDUCTOR=${TORCH_INDUCTOR:-"0"}


    if [[ "0" != ${TORCH_INDUCTOR} ]];then  
        export TORCHINDUCTOR_FREEZING=1
    fi 

    python -m intel_extension_for_pytorch.cpu.launch ${ARGS_IPEX} --log_path=${OUTPUT_DIR} --log_file_prefix="./${LOG_PREFIX}" ${TRAIN_SCRIPT} \
        --input_dir ${DATASET_DIR}/2048_shards_uncompressed_128/ \
        --eval_dir ${DATASET_DIR}/eval_set_uncompressed/ \
        --model_type 'bert' \
        --ipex \
        --output_dir $OUTPUT_DIR/model_save \
        --dense_seq_output \
        --config_name ${BERT_MODEL_CONFIG} \
        $ARGS \
        $params
        2>&1 | tee ${OUTPUT_DIR}/${LOG_PREFIX}
        wait 
else
    rm -rf ${OUTPUT_DIR}/throughput_log_phase2_*
    LOG_PREFIX="throughput_log_phase2_${precision}"
    
    PRETRAINED_MODEL=${PRETRAINED_MODEL-${PWD}/checkpoint_phase1_dir}

    TORCH_INDUCTOR=${TORCH_INDUCTOR:-"0"}   

    params="$params --phase2"

    if [[ "0" != ${TORCH_INDUCTOR} ]];then  
        export TORCHINDUCTOR_FREEZING=1
    fi 
    python -m intel_extension_for_pytorch.cpu.launch ${ARGS_IPEX} --log_path=${OUTPUT_DIR} --log_file_prefix="./${LOG_PREFIX}" ${TRAIN_SCRIPT} \
        --input_dir ${DATASET_DIR}/2048_shards_uncompressed_512/ \
        --eval_dir ${DATASET_DIR}/eval_set_uncompressed/ \
        --model_type 'bert' \
        --model_name_or_path ${PRETRAINED_MODEL} \
        --benchmark \
        --ipex \
        --dense_seq_output \
        --output_dir $OUTPUT_DIR/model_save \
        $ARGS \
        $params
        2>&1 | tee ${OUTPUT_DIR}/${LOG_PREFIX}
        wait 
fi

total_throughput=0
total_latency=0 
total_accuracy=0
num_logs=0

for log_file in ${OUTPUT_DIR}/${LOG_PREFIX}*; do
    throughput=$(grep -oP "Throughput: \K\d+\.\d+" $log_file)
    if [ -z "$throughput" ]; then
        continue  
    latency=$(grep -oP "bert_train latency: \K\d+\.\d+" ${log_file})
    accuracy=$(grep -oP "final_mlm_accuracy: \K\d+\.\d+" ${log_file})
        
    total_throughput=$(bc <<< "$total_throughput + $throughput")
    total_latency=$(bc <<< "$total_latency + $latency")
    total_accuracy=$(bc <<< "$total_accuracy + $accuracy")
    ((num_logs++))
done

if [ $num_logs -gt 0 ]; then
    average_throughput=$(bc <<< "scale=3; $total_throughput / $num_logs")
    average_latency=$(bc <<< "scale=3; $total_latency / $num_logs")
    average_accuracy=$(bc <<< "scale=3; $total_accuracy / $num_logs")

    echo "Average throughput across all valid logs: $average_throughput examples per second" | tee -a ${OUTPUT_DIR}/${LOG_PREFIX}_summary.log
    echo "Average latency across all valid logs: $average_latency seconds per example" | tee -a ${OUTPUT_DIR}/${LOG_PREFIX}_summary.log
    echo "Average accuracy across all valid logs: $average_accuracy %" | tee -a ${OUTPUT_DIR}/${LOG_PREFIX}_summary.log

else
    echo "No valid throughput/accuracy logs found for calculation." | tee -a ${OUTPUT_DIR}/${LOG_PREFIX}_summary.log
    exit
fi

yaml_content=$(cat << EOF
results: 
- key : throughput
  value: $average_throughput
  unit: sentence/s 
- key: latency
  value: $average_latency
  unit: s
- key: accuracy
  value: $average_accuracy
  unit: f1
EOF
)

echo "$yaml_content" >  $OUTPUT_DIR/${LOG_PREFIX}_results.yaml
echo "YAML file created."
