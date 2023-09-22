#!/usr/bin/env bash
#
# Copyright (c) 2022 Intel Corporation
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
echo 'PRECISION='$PRECISION
echo 'OUTPUT_DIR='$OUTPUT_DIR
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

export TF_NUM_INTEROP_THREADS=1

# Create an array of input directories that are expected and then verify that they exist
declare -A input_envs
input_envs[PRECISION]=${PRECISION}
input_envs[OUTPUT_DIR]=${OUTPUT_DIR}
input_envs[GPU_TYPE]=${GPU_TYPE}

for i in "${!input_envs[@]}"; do
  var_name=$i
  env_param=${input_envs[$i]}
 
  if [[ -z $env_param ]]; then
    echo "The required environment variable $var_name is not set" >&2
    exit 1
  fi
done

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}
declare -a str

# If batch size env is not mentioned, then the workload will run with the default batch size.
if [ -z "${BATCH_SIZE}"]; then
  BATCH_SIZE="1024"
  echo "Running with default batch size of ${BATCH_SIZE}"
fi

# Check for GPU type
if [[ $GPU_TYPE == "flex_series" ]]; then
  export OverrideDefaultFP64Settings=1 
  export IGC_EnableDPEmulation=1 
  export TF_NUM_INTEROP_THREADS=1
  export CFESingleSliceDispatchCCSMode=1
  if [[ $PRECISION == "int8" ]]; then
    echo "Precision is $PRECISION"
    if [[ ! -f "${FROZEN_GRAPH}" ]]; then
      pretrained_model=/workspace/tf-flex-series-resnet50v1-5-inference/pretrained_models/resnet50v1_5-frozen_graph-${PRECISION}-gpu.pb
    else
      pretrained_model=${FROZEN_GRAPH}
    fi
  else 
    echo "FLEX SERIES GPU SUPPORTS ONLY INT8 PRECISION"
    exit 1
  fi
elif [[ $GPU_TYPE == "max_series" ]]; then
  if [[ $PRECISION == "int8" || $PRECISION == "fp16" || $PRECISION == "fp32" ]]; then
    echo "Precision is $PRECISION"
    if [[ ! -f "${FROZEN_GRAPH}" ]]; then
      pretrained_model=/workspace/tf-max-series-resnet50v1-5-inference/pretrained_models/resnet50v1_5-frozen_graph-${PRECISION}-gpu.pb
    else
      pretrained_model=${FROZEN_GRAPH}
    fi
  else 
    echo "MAX SERIES GPU SUPPORTS ONLY INT8, FP32 AND FP16 PRECISION"
    exit 1
  fi
fi

if [[ $PRECISION == "fp16" ]]; then
  DTYPE="float16"
  export ITEX_AUTO_MIXED_PRECISION=1
  export ITEX_AUTO_MIXED_PRECISION_DATA_TYPE="FLOAT16"
fi

if [[ $PRECISION == "int8" ]]; then
  DTYPE="int8"
  benchmark="--benchmark"
else
  benchmark=""
fi

if [[ $PRECISION == "fp32" ]]; then
  DTYPE="float32"
fi

if [[ -z "${NUM_OAM}" ]]; then
    echo "resnet50 v1.5 int8 inference"
         mac=`lspci | grep Dis| head -n 1| awk '{print $1}'`
         node=`lspci -s $mac -v | grep NUMA | awk -F, '{print $5}' | awk '{print $3}'`
         numactl -N $node -l python -u models/image_recognition/tensorflow/resnet50v1_5/inference/gpu/int8/eval_image_classifier_inference.py \
         --input-graph=${pretrained_model} \
         --warmup-steps=10 \
         --steps=500 \
         --batch-size=${BATCH_SIZE} \
         --benchmark 
elif [[ ${NUM_OAM} == "4" ]]; then
        echo "resnet50 v1.5 int8 inference on ${NUM_OAM} OAM modules"
        NUM_TILES_PER_GPU=2
        for i in $( eval echo {0..$((NUM_OAM-1))} )
          do
            for j in $( eval echo {0..$((NUM_TILES_PER_GPU-1))} )
                do
                  str+=("ZE_AFFINITY_MASK="${i}"."${j}" python -u models/image_recognition/tensorflow/resnet50v1_5/inference/gpu/int8/eval_image_classifier_inference.py \
                        --input-graph=${pretrained_model} \
                        --warmup-steps=5 \
                        --steps=20 \
                        --batch-size=${BATCH_SIZE} \
                        --dtype ${DTYPE} \
                        ${benchmark} 2>&1 | tee ${OUTPUT_DIR}/resnet50_inf_${PRECISION}_c${i}_t${j}_raw.log & ")
                done
          done
        str=${str[@]}
        cmd_line=${str::-2}
        eval $cmd_line
else
    echo "Currently only x4 OAM Modules are supported"
    exit 1
fi
