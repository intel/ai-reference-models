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

MODEL_DIR=${MODEL_DIR-$PWD}

echo 'MODEL_DIR='$MODEL_DIR
echo 'PRECISION='$PRECISION
echo 'OUTPUT_DIR='$OUTPUT_DIR
echo 'GPU_TYPE='$GPU_TYPE

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

export TF_NUM_INTEROP_THREADS=1
export ITEX_LAYOUT_OPT=1

# Create an array of input directories that are expected and then verify that they exist
declare -A input_envs
input_envs[OUTPUT_DIR]=${OUTPUT_DIR}
input_envs[MODEL_NAME]=${MODEL_NAME}
input_envs[IMAGE_FILE]=${IMAGE_FILE}
input_envs[GPU_TYPE]=${GPU_TYPE}
input_envs[PRECISION]=${PRECISION}

for i in "${!input_envs[@]}"; do
  var_name=$i
  env_param=${input_envs[$i]}
 
  if [[ -z $env_param ]]; then
    echo "The required environment variable $var_name is not set" >&2
    exit 1
  fi
done

mkdir -p ${OUTPUT_DIR}

# If batch size env is not mentioned, then the workload will run with the default batch size.
if [ -z "${BATCH_SIZE}" ]; then
  BATCH_SIZE="64"
  echo "Running with default batch size of ${BATCH_SIZE}"
fi

declare -a str
device_id=$( lspci | grep -i display | sed -n '1p' | awk '{print $7}' )
num_devs=$(lspci | grep -i display | awk '{print $7}' | wc -l)
num_threads=1
k=0

if [[ "$PRECISION" == "fp16" ]];then
  if [[ "$GPU_TYPE" == flex_170 ]]; then
    if [[ ${device_id} == "56c0" ]]; then 
      echo "${MODEL_NAME} FP16 inference on Flex 170"
      python -u $MODEL_DIR/models/image_recognition/tensorflow/efficientnet/inference/gpu/predict.py git-m ${MODEL_NAME} -b ${BATCH_SIZE} -i ${IMAGE_FILE} 2>&1 | tee $OUTPUT_DIR/${MODEL_NAME}__xpu_inf_${BATCH_SIZE}.log
    fi
  elif [[ "$GPU_TYPE" == flex_140 ]]; then
    if [[ ${device_id} == "56c1" ]]; then 
      if [[ $BATCH_SIZE == 1 ]]; then
        echo "${MODEL_NAME} FP16 inference with BATCH_SIZE 1 on Flex 140"
        for i in $( eval echo {0..$((num_devs-1))} )
        do
          for j in $( eval echo {1..$num_threads} )
            do
              str+=("ZE_AFFINITY_MASK="${i}" numactl -C ${k} -l python -u $MODEL_DIR/models/image_recognition/tensorflow/efficientnet/inference/gpu/predict.py -m ${MODEL_NAME} -b ${BATCH_SIZE} -i ${IMAGE_FILE} ")
            ((k=k+1))
            done
          done
      parallel --lb -d, --tagstring "[{#}]" ::: "${str[@]}" 2>&1 | tee $OUTPUT_DIR/${MODEL_NAME}__xpu_inf_c0_c1_${BATCH_SIZE}.log
    else
      echo "${MODEL_NAME} FP16 inference with BATCH_SIZE $BATCH_SIZE on Flex 140"
      for i in $( eval echo {0..$((num_devs-1))} )
        do
          str+=("ZE_AFFINITY_MASK="${i}" python $MODEL_DIR/models/image_recognition/tensorflow/efficientnet/inference/gpu/predict.py \
          -m ${MODEL_NAME} -b ${BATCH_SIZE} \
          -i ${IMAGE_FILE} ")
        done
      parallel --lb -d, --tagstring "[{#}]" ::: "${str[@]}" 2>&1 | tee $OUTPUT_DIR/${MODEL_NAME}__xpu_inf_c0_c1_${BATCH_SIZE}.log
    fi
    file_loc=$OUTPUT_DIR/${MODEL_NAME}__xpu_inf_c0_c1_${BATCH_SIZE}.log
    total_throughput=$( cat $file_loc | grep Throughput | awk '{print $3}' | awk '{ sum_total += $1 } END { print sum_total }' )
    echo 'Total Throughput in images/sec: '$total_throughput | tee -a $file_loc
    fi
  fi
else 
  echo "Efficient Net currently supports FP16 precision"
  exit 1
fi
