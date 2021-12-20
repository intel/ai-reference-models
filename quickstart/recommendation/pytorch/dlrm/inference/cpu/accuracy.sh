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

MODEL_DIR=${MODEL_DIR-$PWD}
if [ ! -e "${MODEL_DIR}/models/recommendation/pytorch/dlrm/product/dlrm_s_pytorch.py"  ]; then
    echo "Could not find the script of dlrm_s_pytorch.py. Please set environment variable '\${MODEL_DIR}'."
    echo "From which the dlrm_s_pytorch.py exist at the: \${MODEL_DIR}/models/recommendation/pytorch/dlrm/product/dlrm_s_pytorch.py"
    exit 1
fi

MODEL_SCRIPT=${MODEL_DIR}/models/recommendation/pytorch/dlrm/product/dlrm_s_pytorch.py

echo "PRECISION: ${PRECISION}"
echo "DATASET_DIR: ${DATASET_DIR}"
echo "WEIGHT_PATH: ${WEIGHT_PATH}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

if [ -z "${DATASET_DIR}" ]; then
  echo "The required environment variable DATASET_DIR has not been set"
  exit 1
fi

if [ ! -d "${DATASET_DIR}" ]; then
  echo "The DATASET_DIR '${DATASET_DIR}' does not exist"
  exit 1
fi

if [ -z "${WEIGHT_PATH}" ]; then
  echo "The required environment variable WEIGHT_PATH has not been set"
  exit 1
fi

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}/dlrm_inference_accuracy_log

LOG=${OUTPUT_DIR}/dlrm_inference_accuracy_log/${PRECISION}

if [[ "$PRECISION" == *"avx"* ]]; then
    unset DNNL_MAX_CPU_ISA
fi

ARGS=""
if [[ $PRECISION == "int8" || $PRECISION == "avx-int8" ]]; then
    echo "running int8 path"
    ARGS="$ARGS --int8 --int8-configure=${MODEL_DIR}/models/recommendation/pytorch/dlrm/product/int8_configure.json"
elif [[ $PRECISION == "bf16" ]]; then
    ARGS="$ARGS --bf16"
    echo "running bf16 path"
elif [[ $PRECISION == "fp32" || $PRECISION == "avx-fp32" ]]; then
    echo "running fp32 path"
else
    echo "The specified PRECISION '${PRECISION}' is unsupported."
    echo "Supported PRECISIONs are: fp32, avx-fp32, bf16, int8, and avx-int8"
    exit 1
fi

CORES=`lscpu | grep Core | awk '{print $4}'`
# use first socket
numa_cmd="numactl -C 0-$((CORES-1))  "
echo "will run on core 0-$((CORES-1)) on socket 0" 

export OMP_NUM_THREADS=$CORES
$numa_cmd python -u $MODEL_SCRIPT \
--raw-data-file=${DATASET_DIR}/day --processed-data-file=${DATASET_DIR}/terabyte_processed.npz \
--data-set=terabyte \
--memory-map --mlperf-bin-loader --round-targets=True --learning-rate=1.0 \
--arch-mlp-bot=13-512-256-128 --arch-mlp-top=1024-1024-512-256-1 \
--arch-sparse-feature-size=128 --max-ind-range=40000000 \
--numpy-rand-seed=727  --inference-only --ipex-interaction \
--print-freq=100 --print-time --mini-batch-size=2048 --test-mini-batch-size=16384 \
--test-freq=2048 --print-auc $ARGS \
--load-model=${WEIGHT_PATH} | tee $LOG

accuracy=$(grep 'Accuracy:' $LOG |sed -e 's/.*Accuracy//;s/[^0-9.]//g')
echo ""dlrm";"auc";${PRECISION};16384;${accuracy}" | tee -a ${OUTPUT_DIR}/summary.log
