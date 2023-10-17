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
echo 'BATCH_SIZE='$BATCH_SIZE
echo 'GPU_TYPE='$GPU_TYPE

# Create an array of input directories that are expected and then verify that they exist
declare -A input_envs
input_envs[PRECISION]=${PRECISION}
input_envs[OUTPUT_DIR]=${OUTPUT_DIR}
input_envs[BATCH_SIZE]=${BATCH_SIZE}
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


if [[ ! -d ${MODEL_DIR}/DeepLearningExamples ]]; then
    echo "ERROR:https://github.com/NVIDIA/DeepLearningExamples.git repo is not cloned and patched. Please clone the repo and apply the patch"
    exit 1
fi

declare -a str
device_id=$( lspci | grep -i display | sed -n '1p' | awk '{print $7}' )
num_devs=$(lspci | grep -i display | awk '{print $7}' | wc -l)
num_threads=1
k=0

#Download pre-trained model
mkdir -p ${MODEL_DIR}/pretrained_weights
python -u ${MODEL_DIR}/DeepLearningExamples/TensorFlow2/Segmentation/MaskRCNN/scripts/download_weights.py --save_dir=${MODEL_DIR}/pretrained_weights

if [ ${PRECISION} == "fp16" ]; then
  if [[ ${GPU_TYPE} == flex_170 ]]; then 
    if [[ ${device_id} == "56c0" ]]; then 
      echo "Running ${PRECISION} MaskRCNN Inference on Flex 170"
      python -u ${MODEL_DIR}/DeepLearningExamples/TensorFlow2/Segmentation/MaskRCNN/scripts/inference.py \ 
      --data_dir=$DATASET_DIR --batch_size=$BATCH_SIZE \
      --no_xla --weights_dir=${MODEL_DIR}/pretrained_weights \
      --amp 2>&1 | tee $OUTPUT_DIR/maskrcnn__xpu_inf_${BATCH_SIZE}.log
    fi
  elif [[ ${GPU_TYPE} == flex_140 ]]; then 
    if [[ ${device_id} == "56c1" ]]; then
      if [[ ${BATCH_SIZE} == 1 ]]; then 
        echo "Running ${PRECISION} MaskRCNN Inference with BATCH SIZE 1 on Flex 140"
        for i in $( eval echo {0..$((num_devs-1))} )
          do
            for j in $( eval echo {1..$num_threads} )
            do
            str+=("ZE_AFFINITY_MASK="${i}" numactl -C ${k} -l \
            python -u ${MODEL_DIR}/DeepLearningExamples/TensorFlow2/Segmentation/MaskRCNN/scripts/inference.py --data_dir=$DATASET_DIR --batch_size=$BATCH_SIZE --no_xla --weights_dir=${MODEL_DIR}/pretrained_weights --amp ")
            ((k=k+1))
            done
        done
        parallel --lb -d, --tagstring "[{#}]" ::: "${str[@]}" 2>&1 | tee $OUTPUT_DIR/maskrcnn__xpu_inf_c0_c1_${BATCH_SIZE}.log
      else 
        echo "Running ${PRECISION} MaskRCNN Inference with BATCH SIZE $BATCH_SIZE on Flex 140"
        for i in $( eval echo {0..$((num_devs-1))} )
        do
          str+=("ZE_AFFINITY_MASK="${i}" python ${MODEL_DIR}/DeepLearningExamples/TensorFlow2/Segmentation/MaskRCNN/scripts/inference.py \
          --data_dir=$DATASET_DIR --batch_size=$BATCH_SIZE \
          --no_xla --weights_dir=${MODEL_DIR}/pretrained_weights --amp ")
        done
        parallel --lb -d, --tagstring "[{#}]" ::: "${str[@]}" 2>&1 | tee $OUTPUT_DIR/maskrcnn__xpu_inf_c0_c1_${BATCH_SIZE}.log
      fi
    file_loc=$OUTPUT_DIR/maskrcnn__xpu_inf_c0_c1_${BATCH_SIZE}.log
    total_predict_throughput=$( grep 'step()' $file_loc | grep 'predict_throughput' | tail -2 | awk '{print $8}' | cut -d , -f 1 | awk '{ sum_total += $1 } END { print sum_total }')
    echo 'Total Throughput: '$total_predict_throughput | tee -a $file_loc
    fi
  fi
else
    echo "MaskRCNN Inference currently supports FP16 inference"
    exit 1
fi
