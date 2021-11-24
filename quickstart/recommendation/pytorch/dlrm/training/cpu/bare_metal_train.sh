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

if [[ "$NUM_BATCH" != "" ]]
then
    ARGS="$ARGS --num-batches=${NUM_BATCH}"
    echo "will early stop after ${NUM_BATCH} batches"
fi

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}
LOG=${OUTPUT_DIR}/dlrm_training_log/${PRECISION}
rm -rf ${LOG}
mkdir -p ${LOG}

ARGS=""
if [[ $PRECISION == "bf16" ]]; then
    ARGS="$ARGS --bf16"
    echo "running bf16 path"
elif [[ $PRECISION == "fp32" ]]; then
    echo "running fp32 path"
else
    echo "The specified PRECISION '${PRECISION}' is unsupported."
    echo "Supported PRECISIONs are: fp32, bf16"
    exit 1
fi

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
BATCHSIZE=$((128*CORES))
export OMP_NUM_THREADS=$CORES
for i in $(seq 1 $((SOCKETS-1))); do
  LOG_i="${LOG}/socket_$i"
  start=$((i*CORES))
  end=$((start+CORES-1))
  numa_cmd="numactl -C $start-$end -m $i"
  $numa_cmd python -u $MODEL_SCRIPT \
  --raw-data-file=${DATASET_DIR}/day --processed-data-file=${DATASET_DIR}/terabyte_processed.npz \
  --data-set=terabyte \
  --memory-map --mlperf-bin-loader --round-targets=True --learning-rate=1.0 \
  --arch-mlp-bot=13-512-256-128 --arch-mlp-top=1024-1024-512-256-1 \
  --arch-sparse-feature-size=128 --max-ind-range=40000000 \
  --numpy-rand-seed=727 --print-auc --mlperf-auc-threshold=0.8025 \
  --mini-batch-size=${BATCHSIZE} --print-freq=100 --print-time --ipex-interaction \
  --test-freq=5000 --test-mini-batch-size=16384 \
  $ARGS |tee $LOG_i &
done

i=0
LOG_0="${LOG}/socket_$i"
start=$((i*CORES))
end=$((start+CORES-1))
numa_cmd="numactl -C $start-$end -m $i"
$numa_cmd python -u $MODEL_SCRIPT \
  --raw-data-file=${DATASET_DIR}/day --processed-data-file=${DATASET_DIR}/terabyte_processed.npz \
  --data-set=terabyte \
  --memory-map --mlperf-bin-loader --round-targets=True --learning-rate=1.0 \
  --arch-mlp-bot=13-512-256-128 --arch-mlp-top=1024-1024-512-256-1 \
  --arch-sparse-feature-size=128 --max-ind-range=40000000 \
  --numpy-rand-seed=727 --print-auc --mlperf-auc-threshold=0.8025 \
  --mini-batch-size=${BATCHSIZE} --print-freq=100 --print-time --ipex-interaction \
  --test-freq=5000 --test-mini-batch-size=16384 \
  $ARGS |tee $LOG_0
wait

throughput=$(grep 'Throughput:' ${LOG}/socket* |sed -e 's/.*Throughput//;s/[^0-9.]//g' |awk '
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
echo ""dlrm";"throughput";${PRECISION};${BATCHSIZE};${throughput}" | tee -a ${OUTPUT_DIR}/summary.log
