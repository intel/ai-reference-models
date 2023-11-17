#!/bin/bash
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
# ============================================================================

MODEL_DIR=${MODEL_DIR-$PWD}
GPJ_LOG_LEVEL=${GPJ_LOG_LEVEL-10}
WARMUP=${WARMUP-false}
PROFILE=${PROFILE-false}
OPT=${OPT-true}
SORT=${SORT-true}
BEAM=${BEAM-4}
WORLD_SIZE=${WORLD_SIZE-8}
START=${START-0}
MAX_EXP=${MAX_EXP--1}
PAD=${PAD-left}
ENABLE_SDP_FUSION=${ENABLE_SDP_FUSION-1}  
DISABLE_KV_CACHE=${DISABLE_KV_CACHE-0}
DYNAMIC_BATCHING=${DYNAMIC_BATCHING-false}
ACCURACY=${ACCURACY-true}

declare -A input_envs
input_envs[PRECISION]=${PRECISION}
input_envs[OUTPUT_DIR]=${OUTPUT_DIR}
input_envs[INFERENCE_MODE]=${INFERENCE_MODE}
input_envs[INFERENCE_TYPE]=${INFERENCE_TYPE}
input_envs[DATASET_DIR]=${DATASET_DIR}
input_envs[PRETRAINED_MODEL_DIR]=${PRETRAINED_MODEL_DIR}
input_envs[CONFIG_DIR]=${CONFIG_DIR}
input_envs[NUM_GPU_TILES]=${NUM_GPU_TILES}

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

if [[ ${NUM_GPU_TILES} != 8 ]]; then
    echo " GPT-J 6B Workload supports GPU with 8 tiles. Please provide the value 8"
    exit 1
fi
echo "Use TcMalloc memory allocator"
export LD_PRELOAD=/workspace/lib/tcmalloc/lib/libtcmalloc.so:${LD_PRELOAD}

echo "Use Intel OpenMP"
export LD_PRELOAD=/opt/intel/oneapi/compiler/2023.1.0/linux/compiler/lib/intel64_lin/libiomp5.so:${LD_PRELOAD}

echo "Set KMP"
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
export KMP_AFFINITY=granularity=fine,compact,1,0

echo "Set IPEX-XPU runtime env"
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2
export ENABLE_SDP_FUSION=${ENABLE_SDP_FUSION}
export DISABLE_KV_CACHE=${DISABLE_KV_CACHE}

echo "Set File Descriptor Limitation"
FD_MAX=`ulimit -n -H`
ulimit -n $((FD_MAX-1))

cp ${PRETRAINED_MODEL_DIR}/pytorch_model.bin.index.json /workspace/pytorch-max-series-gpt-j-6b-mlperf-inference

if [[ ${PRECISION} == "int4" ]]; then
    if [[ ! -f ${PRETRAINED_MODEL_DIR}/int4_weight_pro.pt ]]; then 
        source /workspace/pytorch-max-series-gpt-j-6b-mlperf-inference/venvs/int4-quantization-env/bin/activate
        echo "Calibrating model..."
        cd ${MODEL_DIR}/models/generative-ai/pytorch/gpt-j-6b-mlperf/gpu
        python -u prepare_calibration.py \
           --calibration-list-file ${CONFIG_DIR}/calibration-list.txt \
           --output-dir ${DATASET_DIR}
        cd -
        git clone https://github.com/qwopqwop200/GPTQ-for-LLaMa.git 
        cp ${MODEL_DIR}/models/generative-ai/pytorch/gpt-j-6b-mlperf/gpu/run_quantization.sh ${MODEL_DIR}/models/generative-ai/pytorch/gpt-j-6b-mlperf/gpu/cnn_dm_dataset.py GPTQ-for-LLaMa
        cd GPTQ-for-LLaMa
        git apply ${MODEL_DIR}/models/generative-ai/pytorch/gpt-j-6b-mlperf/gpu/gptj.patch
        echo "Quantizing model to int4 precision"
        DATA_DIR=${DATASET_DIR} MODEL_DIR=${PRETRAINED_MODEL_DIR}  ./run_quantization.sh
        cd ${MODEL_DIR}/models/generative-ai/pytorch/gpt-j-6b-mlperf/gpu/
        python -u convert_int4_model.py --model_path ${PRETRAINED_MODEL_DIR} --model_name int4_weight_pro.pt
        cd ${MODEL_DIR}
        deactivate 
    fi
    source /workspace/pytorch-max-series-gpt-j-6b-mlperf-inference/venvs/int4-benchmark-env/bin/activate 
    python -u ${MODEL_DIR}/models/generative-ai/pytorch/gpt-j-6b-mlperf/gpu/convert_model_dict.py --model_path ${PRETRAINED_MODEL_DIR}/int4_weight_pro.pt \
           --config_input_path ${PRETRAINED_MODEL_DIR}/pytorch_model.bin.index.json \
           --config_output_path ${PRETRAINED_MODEL_DIR}/pytorch_model.bin.index_int4.json

    cp ${PRETRAINED_MODEL_DIR}/pytorch_model.bin.index_int4.json  ${PRETRAINED_MODEL_DIR}/pytorch_model.bin.index.json

elif [[ ${PRECISION} == "float16" ]]; then
    source /workspace/pytorch-max-series-gpt-j-6b-mlperf-inference/venvs/fp16-benchmark-env/bin/activate
else
    echo "Only float16 and int4 precisions are supported"
    exit 1
fi

SCRIPT_ARGS=" --scenario ${INFERENCE_MODE}"
SCRIPT_ARGS+=" --model_path ${PRETRAINED_MODEL_DIR}"
SCRIPT_ARGS+=" --dataset_path ${DATASET_DIR}/cnn_eval.json"
SCRIPT_ARGS+=" --log_dir ${OUTPUT_DIR}"
SCRIPT_ARGS+=" --device xpu"
SCRIPT_ARGS+=" --dtype ${PRECISION}"
SCRIPT_ARGS+=" --num_beams ${BEAM}"
SCRIPT_ARGS+=" --num_workers ${NUM_GPU_TILES}"
SCRIPT_ARGS+=" --world_size ${WORLD_SIZE}"
SCRIPT_ARGS+=" --start_rank ${START}"
SCRIPT_ARGS+=" --max_examples ${MAX_EXP}"
SCRIPT_ARGS+=" --padding_side ${PAD}"
SCRIPT_ARGS+=" --user_conf ${MODEL_DIR}/models/generative-ai/pytorch/gpt-j-6b-mlperf/gpu/configs/user_${PRECISION}.conf"
SCRIPT_ARGS+=" --mlperf_conf ${CONFIG_DIR}/mlperf.conf"
[ ${WARMUP} == true ] && SCRIPT_ARGS+=" --warmup --warmup_path ${DATASET_DIR}/cnn_eval_warmup.json"
[ ${PROFILE} == true ] && SCRIPT_ARGS+=" --profile"
[ ${ACCURACY} == true ] && SCRIPT_ARGS+=" --accuracy"
[ ${OPT} == true ] && SCRIPT_ARGS+=" --optimize_transformers"
[ ${SORT} == true ] && SCRIPT_ARGS+=" --sort"
if [[ ${INFERENCE_MODE} == "Offline" ]]; then
    BATCH_SIZE=1
    DYNAMIC_BATCHING=true
elif [[ ${INFERENCE_MODE} == "Server" ]]; then
    BATCH_SIZE=${BATCH_SIZE-32}
else
    echo "Only Offline or Server Inference scenarios are supported"
    exit 1
fi
SCRIPT_ARGS+=" --batch_size ${BATCH_SIZE}"
[ ${DYNAMIC_BATCHING} == true ] && SCRIPT_ARGS+=" --dynamic_batching"

if [[ ${INFERENCE_TYPE} == "benchmark" ]]; then
    NUM_GPU_TILES=${NUM_GPU_TILES} python -u ${MODEL_DIR}/models/generative-ai/pytorch/gpt-j-6b-mlperf/gpu/main.py ${SCRIPT_ARGS} 2>&1 | tee ${OUTPUT_DIR}/gpt-j-6b-${PRECISION}-${INFERENCE_MODE}_${INFERENCE_TYPE}_acc${ACCURACY}.log
elif [[ ${INFERENCE_TYPE} == "accuracy" ]]; then
    NUM_GPU_TILES=${NUM_GPU_TILES} python -u ${MODEL_DIR}/models/generative-ai/pytorch/gpt-j-6b-mlperf/gpu/evaluation.py \
    --mlperf_accuracy_file ${OUTPUT_DIR}/mlperf_log_accuracy.json \
    --model_path ${PRETRAINED_MODEL_DIR} \
    --dataset_path ${DATASET_DIR}/cnn_eval.json 2>&1 | tee ${OUTPUT_DIR}/gpt-j-6b-${PRECISION}_${INFERENCE_MODE}_accuracy.log
else 
    echo "Only accuracy and benchmark inference types supported"
    exit 1
fi

cp /workspace/pytorch-max-series-gpt-j-6b-mlperf-inference/pytorch_model.bin.index.json ${PRETRAINED_MODEL_DIR}
