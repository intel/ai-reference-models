#!/usr/bin/env bash
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

MODEL_DIR=${MODEL_DIR-$PWD}

if [ ! -e "${MODEL_DIR}/train.py" ]; then
  echo "Could not find the script of train.py. Please set environment variable '\${MODEL_DIR}'."
  echo "From which the train.py exist at the: \${MODEL_DIR}/train.py"
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

if [ -z "${PRECISION}" ]; then
  echo "The required environment variable PRECISION has not been set"
  echo "Please set PRECISION to fp32, avx-fp32, bf32 or bf16."
  exit 1
fi

MODEL_CONFIG=${5:-"${MODEL_DIR}/rnnt.toml"}
RESULT_DIR=${6:-"${MODEL_DIR}/results"}
CHECKPOINT=${7:-"none"}
CREATE_LOGFILE=${8:-"true"}
CUDNN_BENCHMARK=${9:-"true"}
NUM_GPUS=${10:-0}
PRECISION=${PRECISION:-"fp32"}
EPOCHS=${12:-1}
SEED=${13:-2021}
BATCH_SIZE=${BATCH_SIZE:-64}
EVAL_BATCH_SIZE=${15:-2}
LEARNING_RATE=${16:-"0.001"}
LEARNING_RATE_WARMUP=${17:-"8000"}
GRADIENT_ACCUMULATION_STEPS=${18:-1}
LAUNCH_OPT=${LAUNCH_OPT:-"none"}

if [[ -z "${DISTRIBUTED}" ]]; then
    echo "Running single-node training"
    LOG_PREFIX="training_throughput_log"
else
    echo "Running distributed multi-node training"
    LOG_PREFIX="distributed_training_throughput_log"
    SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
    NNODES=${NNODES:-1}
    HOSTFILE=${HOSTFILE:-"${MODEL_DIR}/hostfile"}
    NUM_RANKS=$(( NNODES * SOCKETS ))
    LOCAL_BATCH_SIZE=$(( BATCH_SIZE / NUM_RANKS))
fi

if [[ $PRECISION == "avx-fp32" ]]; then
    unset DNNL_MAX_CPU_ISA
fi

PREC=""
if [ "$PRECISION" = "bf16" ]; then
    PREC="--bf16"
    precision="bf16"
    echo "### running bf16 datatype"
elif [ "$PRECISION" = "fp32" ] ; then
    PREC="--fp32"
    precision="fp32"
    echo "### running fp32 datatype"
elif [ "$PRECISION" = "bf32" ]; then
    PREC="--bf32"
    precision="bf32"
    echo "### running bf32 datatype"
else
    echo "The specified precision '$PRECISION' is unsupported."
    echo "Supported precisions now are: fp32, avx-fp32, bf16 and bf32"
fi

IPEX="--ipex"

PROFILE=""
if [ "$3" = profiling ]; then
    PROFILE="--profiling"
fi

WARMUP=20

if [ "$CHECKPOINT" = "none" ] ; then
   CHECKPOINT=""
else
   CHECKPOINT=" --ckpt=${CHECKPOINT}"
fi

CMD=" --batch_size=$BATCH_SIZE"
CMD+=" --eval_batch_size=$EVAL_BATCH_SIZE"
CMD+=" --num_epochs=$EPOCHS"
CMD+=" --output_dir=$RESULT_DIR"
CMD+=" --model_toml=$MODEL_CONFIG"
CMD+=" --lr=$LEARNING_RATE"
CMD+=" --lr_warmup=$LEARNING_RATE_WARMUP"
CMD+=" --seed=$SEED"
CMD+=" --optimizer=adam"
CMD+=" --dataset_dir=$DATASET_DIR/dataset/LibriSpeech"
CMD+=" --val_manifest=$DATASET_DIR/dataset/LibriSpeech/librispeech-dev-clean-wav.json"
CMD+=" --train_manifest=$DATASET_DIR/dataset/LibriSpeech/librispeech-train-clean-100-wav.json,$DATASET_DIR/dataset/LibriSpeech/librispeech-train-clean-360-wav.json,$DATASET_DIR/dataset/LibriSpeech/librispeech-train-other-500-wav.json"
CMD+=" --weight_decay=1e-3"
CMD+=" --save_freq=100"
CMD+=" --eval_freq=1"
CMD+=" --train_freq=5"
CMD+=" --lr_decay"
CMD+=" --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS"
CMD+=" $CHECKPOINT"
CMD+=" $PREC"
CMD+=" $IPEX"
CMD+=" --warmup=$WARMUP"
CMD+=" $PROFILE"

# TODO: FP32 is still under development. For current validation,
# in FP32, it only runs 100 iterations. NUM_STEPS is disabled in FP32.
if [[ "${PRECISION}" == "fp32" ]] ; then
    CMD+=" --num_steps=100"
elif [[ ! -z "${NUM_STEPS}" ]]; then
    CMD+=" --num_steps=$NUM_STEPS"
fi

ARGS_IPEX=""

if [[ -z "${DISTRIBUTED}" ]]; then
    export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
    export KMP_BLOCKTIME=1
    export KMP_AFFINITY=granularity=fine,compact,1,0
    ARGS_IPEX="${ARGS_IPEX} --memory-allocator jemalloc --node_id=0"
else
    CMD+=" --backend=ccl"
    oneccl_bindings_for_pytorch_path=$(python -c "import torch; import oneccl_bindings_for_pytorch; import os;  print(os.path.abspath(os.path.dirname(oneccl_bindings_for_pytorch.__file__)))")
    source $oneccl_bindings_for_pytorch_path/env/setvars.sh
    export FI_PROVIDER=psm3
    export PSM3_HAL=sockets
    ARGS_IPEX="${ARGS_IPEX} --distributed --nnodes ${NNODES} --hostfile ${HOSTFILE}"
fi

rm -rf ${OUTPUT_DIR}/${LOG_PREFIX}*

python -m intel_extension_for_pytorch.cpu.launch \
    ${ARGS_IPEX} \
    ${MODEL_DIR}/train.py ${CMD} \
    2>&1 | tee ${OUTPUT_DIR}/${LOG_PREFIX}_${precision}.txt

wait

throughput=$(grep 'Throughput:' ${OUTPUT_DIR}/${LOG_PREFIX}* |sed -e 's/.*Throughput//;s/[^0-9.]//g' |awk '
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
echo "--------------------------------Performance Summary per NUMA Node--------------------------------"
echo ""RNN-T";"training throughput";${precision};${BATCH_SIZE};${throughput}" | tee -a ${OUTPUT_DIR}/summary.log

accuracy="0"
latency="0"

yaml_content=$(cat << EOF
results:
- key : throughput
  value: $throughput
  unit: fps
- key: latency
  value: $latency
  unit: ms
- key: accuracy
  value: $accuracy
  unit: AP
EOF
)

echo "$yaml_content" >  $OUTPUT_DIR/results.yaml
echo "YAML file created."
