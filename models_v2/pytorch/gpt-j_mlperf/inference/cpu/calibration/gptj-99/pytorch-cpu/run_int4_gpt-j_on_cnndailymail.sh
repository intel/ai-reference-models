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
set -x

export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}

export GROUP_SIZE=-1
export OUTPUT_DIR=${1:-saved_results}

if [ -e "${INT4_MODEL_DIR}/best_int4_model.pt" ]; then
    echo No Quantization
    exit 0
fi
# Unset a few env variables because they slow down GPTQ calibration
unset LD_PRELOAD \
      KMP_BLOCKTIME \
      KMP_TPAUSE \
      KMP_SETTINGS \
      KMP_AFFINITY \
      KMP_FORKJOIN_BARRIER_PATTERN \
      KMP_PLAIN_BARRIER_PATTERN \
      KMP_REDUCTION_BARRIER_PATTERN

# Download finetuned GPT-J model
echo "`date +%Y-%m-%d\ %T` - INFO - Download finetuned GPT-J model..."
#wget https://cloud.mlcommons.org/index.php/s/QAZ2oM94MkFtbQx/download -O gpt-j-checkpoint.zip
retVal=$?
if [ $retVal -ne 0 ]; then
    echo "`date +%Y-%m-%d\ %T` - ERROR - Downloading finetuned GPT-J model failed. Exit."
    exit $retVal
fi
echo "`date +%Y-%m-%d\ %T` - INFO - Finetuned GPT-J model downloaded"
echo "`date +%Y-%m-%d\ %T` - INFO - Extract GPT-J model..."
#unzip -q gpt-j-checkpoint.zip
model_path=${CHECKPOINT_DIR}
echo "`date +%Y-%m-%d\ %T` - INFO - GPT-J model extracted to  ${model_path}"
# Run GPTQ calibration

python ./run_int4_gpt-j_on_cnndailymail.py \
    --model ${model_path} \
    --output-dir ${OUTPUT_DIR} \
    --group-size ${GROUP_SIZE} 

retVal=$?
if [ $retVal -ne 0 ]; then
    echo "`date +%Y-%m-%d\ %T` - ERROR - Exit."
    exit $retVal
fi

# Set a few env variables to get best performance
export LD_PRELOAD=${CONDA_PREFIX}/lib/libstdc++.so.6
export KMP_BLOCKTIME=INF
export KMP_TPAUSE=0
export KMP_SETTINGS=1
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_FORKJOIN_BARRIER_PATTERN=dist,dist
export KMP_PLAIN_BARRIER_PATTERN=dist,dist
export KMP_REDUCTION_BARRIER_PATTERN=dist,dist
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so # Intel OpenMP
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so

# Run benchmark
python ./run_int4_gpt-j_on_cnndailymail.py \
    --dataset-path ./saved_results/cnn_dailymail_validation.json \
    --model ${model_path} \
    --output-dir ${OUTPUT_DIR} \
    --low-precision-checkpoint ${OUTPUT_DIR}/gptq_checkpoint_g${GROUP_SIZE}.pt
retVal=$?
if [ $retVal -ne 0 ]; then
    echo "`date +%Y-%m-%d\ %T` - ERROR - Exit."
    exit $retVal
fi
echo "`date +%Y-%m-%d\ %T` - INFO - Finished successfully."

rm -rf ${INT4_MODEL_DIR} && mkdir -p ${INT4_MODEL_DIR} && ln -sf ${CALIBRATION_DIR}/saved_results/int4_model.pt ${INT4_MODEL_DIR}/best_int4_model.pt 2>&1 | tee ${CONTAINER_OUTPUT_DIR}/preproc_${MODEL}_${IMPL}_${DTYPE}.log

set +x
