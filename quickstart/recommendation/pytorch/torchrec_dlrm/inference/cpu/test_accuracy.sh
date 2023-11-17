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
if [ ! -e "${MODEL_DIR}/models/recommendation/pytorch/torchrec_dlrm/dlrm_main.py"  ]; then
    echo "Could not find the script of dlrm_s_pytorch.py. Please set environment variable '\${MODEL_DIR}'."
    echo "From which the dlrm_s_pytorch.py exist at the: \${MODEL_DIR}/models/recommendation/pytorch/torchrec_dlrm/dlrm_main.py"
    exit 1
fi

export MODEL_SCRIPT=${MODEL_DIR}/models/recommendation/pytorch/torchrec_dlrm/dlrm_main.py
export INT8_CONFIG=${MODEL_DIR}/models/recommendation/pytorch/torchrec_dlrm/int8_configure.json
export BATCH_SIZE=65536

echo "PRECISION: ${PRECISION}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

if [ -z "${DATASET_DIR}" ]; then
  echo "The required environment variable DATASET_DIR has not been set"
  exit 1
fi

if [ -z "${WEIGHT_DIR}" ]; then
  echo "The required environment variable WEIGHT_DIR has not been set"
  exit 1
fi

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}
LOG=${OUTPUT_DIR}/dlrm_inference_accuarcy_log/${PRECISION}
rm -rf ${LOG}
mkdir -p ${LOG}

ARGS=""
export EXTRA_ARGS=" --synthetic_multi_hot_criteo_path $DATASET_DIR --test_auroc --snapshot-dir $WEIGHT_DIR"
if [[ $PRECISION == "bf16" ]]; then
    ARGS="$ARGS --dtype bf16 --ipex-merged-emb-cat"
    echo "running bf16 path"
elif [[ $PRECISION == "fp32" ]]; then
    echo "running fp32 path"
    ARGS="$ARGS --dtype fp32 --ipex-merged-emb-cat"
elif [[ $PRECISION == "bf32" ]]; then
    echo "running bf32 path"
    ARGS="$ARGS --dtype bf32 --ipex-merged-emb-cat"
elif [[ $PRECISION == "fp16" ]]; then
    echo "running fp16 path"
    ARGS="$ARGS --dtype fp16 --ipex-merged-emb-cat"
elif [[ $PRECISION == "int8" ]]; then
    echo "prepare int8 weight"
    bash ${MODEL_DIR}/quickstart/recommendation/pytorch/torchrec_dlrm/inference/cpu/prepare_int8.sh
    echo "running int8 path"
    ARGS="$ARGS --dtype int8 --ipex-merged-emb-cat --int8-configure-dir ${INT8_CONFIG}"
else
    echo "The specified PRECISION '${PRECISION}' is unsupported."
    echo "Supported PRECISIONs are: fp32, bf32, fp16, bf16, int8"
    exit 1
fi

LOG_0="${LOG}/acc.log"

if [[ $PLOTMEM == "true" ]]; then
pip install memory_profiler
export mrun_cmd="mprof run --python -o ${MEMLOG}"
fi

# Do not need to use launcher to bind memory for accuracy test
$mrun_cmd python $MODEL_SCRIPT \
    --snapshot-dir $WEIGHT_DIR \
    --embedding_dim 128 \
    --dense_arch_layer_sizes 512,256,128 \
    --over_arch_layer_sizes 1024,1024,512,256,1 \
    --num_embeddings_per_feature 40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36 \
    --epochs 1 \
    --pin_memory \
    --mmap_mode \
    --batch_size $BATCH_SIZE \
    --interaction_type=dcn \
    --dcn_num_layers=3 \
    --dcn_low_rank_dim=512 \
    --ipex-optimize \
    --inference-only \
    $ARGS $EXTRA_ARGS 2>&1 | tee $LOG_0

if [[ $PLOTMEM == "true" ]]; then
mprof plot ${MEMLOG} -o ${MEMPIC}
fi

accuracy=$(grep 'Final AUROC:' $LOG_0 |sed -e 's/.*Final AUROC//;s/[^0-9.]//g')
echo ""dlrm-v2";"auc";${PRECISION};${BATCH_SIZE};${accuracy}" | tee -a ${OUTPUT_DIR}/summary.log
