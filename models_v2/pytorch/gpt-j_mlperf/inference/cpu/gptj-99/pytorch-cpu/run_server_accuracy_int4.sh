#!/bin/bash
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

export KMP_BLOCKTIME=1
export KMP_AFFINITY=granularity=fine,compact,1,0
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so

export num_physical_cores=`lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l`
num_numa=$(numactl --hardware|grep available|awk -F' ' '{ print $2 }')

python ../../user_config.py user_default_int4.conf user_int4.conf
cat user_int4.conf

NUM_PROC=$num_numa
CPUS_PER_PROC=$((num_physical_cores/num_numa))
WORKERS_PER_PROC=$(((4 * CPUS_PER_PROC) / 64 ))
TOTAL_SAMPLE_COUNT=13368
BATCH_SIZE=1
TIMESTAMP=$(date +%m-%d-%H-%M)
HOSTNAME=$(hostname)
OUTPUT_DIR=server-accuracy-output-${HOSTNAME}-batch-${BATCH_SIZE}-procs-${NUM_PROC}-ins-per-proc-${WORKERS_PER_PROC}-${TIMESTAMP}

LOGICAL_CORES_START=$num_physical_cores

echo python runner.py --workload-name gptj \
        --scenario Server \
        --mode Accuracy \
        --num-proc ${NUM_PROC} \
        --cpus-per-proc ${CPUS_PER_PROC} \
        --model-checkpoint-path ${CHECKPOINT_DIR} \
        --dataset-path ${VALIDATION_DATA_JSON} \
        --batch-size ${BATCH_SIZE} \
        --mlperf-conf mlperf.conf \
        --user-conf user_int4.conf \
        --precision int4_bf16_mixed \
        --warmup \
        --bind-logical-cores \
        --logical-cores-start ${LOGICAL_CORES_START} \
        --quantized-model ${INT4_MODEL_DIR}/best_int4_model.pt \
        --workers-per-proc ${WORKERS_PER_PROC} \
        --total-sample-count ${TOTAL_SAMPLE_COUNT} \
        --output-dir ${OUTPUT_DIR} \
        2>&1 | tee ${OUTPUT_DIR}.log

python runner.py --workload-name gptj \
        --scenario Server \
        --mode Accuracy \
        --num-proc ${NUM_PROC} \
        --cpus-per-proc ${CPUS_PER_PROC} \
        --model-checkpoint-path ${CHECKPOINT_DIR} \
        --dataset-path ${VALIDATION_DATA_JSON} \
        --batch-size ${BATCH_SIZE} \
        --mlperf-conf mlperf.conf \
        --user-conf user_int4.conf \
        --precision int4_bf16_mixed \
        --warmup \
        --bind-logical-cores \
        --logical-cores-start ${LOGICAL_CORES_START} \
        --quantized-model ${INT4_MODEL_DIR}/best_int4_model.pt \
        --workers-per-proc ${WORKERS_PER_PROC} \
        --total-sample-count ${TOTAL_SAMPLE_COUNT} \
        --output-dir ${OUTPUT_DIR} \
        2>&1 | tee ${OUTPUT_DIR}.log

if [ -e ${OUTPUT_DIR}/mlperf_log_accuracy.json ]; then
        echo " ==================================="
        echo "         Evaluating Accuracy        "
        echo " ==================================="

        python evaluation.py --mlperf-accuracy-file ${OUTPUT_DIR}/mlperf_log_accuracy.json \
                --dataset-file ${VALIDATION_DATA_JSON} \
                --model-name-or-path ${CHECKPOINT_DIR} 2>&1 | tee -a accuracy-server-${TIMESTAMP}.txt ${OUTPUT_DIR}.log accuracy.txt
fi

cp ${OUTPUT_DIR}/* .

