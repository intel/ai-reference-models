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

if [[ "$TEST_MODE" == "THROUGHPUT" ]]; then
    echo "TEST_MODE set to THROUGHPUT"
    mode=throughput
elif [[ "$TEST_MODE" == "REALTIME" ]]; then
    echo "TEST_MODE set to REALTIME"
    mode=latency
elif [[ "$TEST_MODE" == "ACCURACY" ]]; then
    echo "TEST_MODE set to ACCURACY"
    mode=accuracy
else
    echo "Please set TEST_MODE to THROUGHPUT, REALTIME or ACCURACY"
    exit
fi

if [ ! -e "${MODEL_DIR}/inference.py" ]; then
  echo "Could not find the script of inference.py. Please set environment variable '\${MODEL_DIR}'."
  echo "From which the inference.py exist at the: \${MODEL_DIR}/inference.py"
  exit 1
fi

if [ ! -d "${DATASET_DIR}" ]; then
  echo "The DATASET_DIR \${DATASET_DIR} does not exist"
  exit 1
fi

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

# Set distributed to false
DISTRIBUTED=${DISTRIBUTED:-'false'}

mkdir -p ${OUTPUT_DIR}

if [[ "${PRECISION}" == *"avx"* ]]; then
    unset DNNL_MAX_CPU_ISA
fi

ARGS=""
if [ "${PRECISION}" == "bf16" ]; then
    ARGS="$ARGS --precision=bf16"
    echo "### running bf16 datatype"
elif [ "${PRECISION}" == "fp16" ]; then
    ARGS="$ARGS --precision=fp16"
    echo "### running fp16 datatype"
elif [ "${PRECISION}" == "int8-bf16" ]; then
    ARGS="$ARGS --precision=int8-bf16"
    if [ "${RUN_MODE}" == "ipex-jit" ]; then
        ARGS="$ARGS --configure-dir=conv_and_linear131.json"
    elif [ "${RUN_MODE}" == "compile-inductor" ]; then
        if [ ! -f "${INT8_MODEL}" ]; then
            echo "The required file INT8_MODEL does not exist"
            exit 1
        fi
        ARGS="$ARGS --quantized_model_path=${INT8_MODEL}"
    else
        echo "For int8-bf16 datatype, the specified mode '${RUN_MODE}' is unsupported."
        echo "Supported mode are: ipex-jit, compile-inductor"
        exit 1
    fi
    echo "### running int8-bf16 datatype"
elif [ "${PRECISION}" == "int8-fp32" ]; then
    ARGS="$ARGS --precision=int8-fp32"
    if [ "${RUN_MODE}" == "ipex-jit" ]; then
        ARGS="$ARGS --configure-dir=conv_and_linear131.json"
    elif [ "${RUN_MODE}" == "compile-inductor" ]; then
        if [ ! -f "${INT8_MODEL}" ]; then
            echo "The required file INT8_MODEL does not exist"
            exit 1
        fi
        ARGS="$ARGS --quantized_model_path=${INT8_MODEL}"
    else
        echo "For int8-fp32 datatype, the specified mode '${RUN_MODE}' is unsupported."
        echo "Supported mode are: ipex-jit, compile-inductor"
        exit 1
    fi
    echo "### running int8-fp32 datatype"
elif [ "${PRECISION}" == "bf32" ]; then
    ARGS="$ARGS --precision=bf32"
    echo "### running bf32 datatype"
elif [ "${PRECISION}" == "fp32" ]; then
    echo "### running fp32 datatype"
else
    echo "The specified precision '${PRECISION}' is unsupported."
    echo "Supported precisions are: fp32, bf32, fp16, bf16, int8-bf16, int8-fp32"
    exit 1
fi

TORCH_INDUCTOR=${TORCH_INDUCTOR:-"0"}
if [ "${RUN_MODE}" == "eager" ]; then
    echo "### running eager mode"
elif [ "${RUN_MODE}" == "ipex-jit" ]; then
    ARGS="$ARGS --ipex --jit"
    echo "### running IPEX JIT mode"
elif [ "${RUN_MODE}" == "compile-ipex" ]; then
    ARGS="$ARGS --compile_ipex"
    echo "### running torch.compile with ipex backend"
elif [[ "${RUN_MODE}" == "compile-inductor" || "1" == "${TORCH_INDUCTOR}" ]]; then
    export TORCHINDUCTOR_FREEZING=1
    export TORCHINDUCTOR_CPP_ENABLE_TILING_HEURISTIC=0
    export TORCHINDUCTOR_ENABLE_LINEAR_BINARY_FOLDING=1
    ARGS="$ARGS --compile_inductor"
    echo "### running torch.compile with inductor backend"
else
    echo "The specified mode '${RUN_MODE}' is unsupported."
    echo "Supported mode are: eager, ipex-jit, compile-ipex, compile-inductor"
    exit 1
fi

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export KMP_BLOCKTIME=200
export KMP_AFFINITY=granularity=fine,compact,1,0

if [[ "$TEST_MODE" == "THROUGHPUT" ]]; then
    num_warmup=${num_warmup:-"1"}
    num_iter=${num_iter:-"10"}
    rm -rf ${OUTPUT_DIR}/LCM_${PRECISION}_inference_throughput*
    MODE_ARGS="--throughput-mode"

elif [[ "$TEST_MODE" == "REALTIME" ]]; then
    CORES=`lscpu | grep 'Core(s)' | awk '{print $4}'`
    SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
    NUMAS=`lscpu | grep 'NUMA node(s)' | awk '{print $3}'`
    CORES_PER_NUMA=`expr $CORES \* $SOCKETS / $NUMAS`
    CORES_PER_INSTANCE=4
    export OMP_NUM_THREADS=$CORES_PER_INSTANCE
    NUMBER_INSTANCE=`expr $CORES_PER_NUMA / $CORES_PER_INSTANCE`
    num_warmup=${num_warmup:-"1"}
    num_iter=${num_iter:-"1"}
    rm -rf ${OUTPUT_DIR}/LCM_${PRECISION}_inference_latency*
    MODE_ARGS="--ninstances $NUMAS --instance-idx $NUMBER_INSTANCE"

elif [[ "$TEST_MODE" == "ACCURACY" ]]; then
    if [[ "$DISTRIBUTED" == "false" ]]; then
        num_warmup=${num_warmup:-"1"}
        num_iter=${num_iter-"10"}
        rm -rf ${OUTPUT_DIR}/LCM_${PRECISION}_inference_accuracy*
        rm -rf ${PRECISION}_results
        MODE_ARGS=" "
    else
        CORES=`lscpu | grep Core | awk '{print $4}'`
        SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
        TOTAL_CORES=`expr $CORES \* $SOCKETS`
        NNODES=${NNODES:-1}
        HOSTFILE=${HOSTFILE:-./hostfile}
        NUM_RANKS=$(( NNODES * SOCKETS ))
        if [ ${LOCAL_BATCH_SIZE} ]; then
            GLOBAL_BATCH_SIZE=$(( LOCAL_BATCH_SIZE * NNODES * SOCKETS ))
        fi
        CORES_PER_INSTANCE=$CORES
        export CCL_WORKER_COUNT=8
        export CCL_LOG_LEVEL=info
        export CCL_BF16=avx512bf
        export CCL_ATL_TRANSPORT=ofi
        export CCL_MNIC_COUNT=2
        export CCL_MNIC=local
        export CCL_MNIC_NAME=irdma1,irdma5
        export CCL_ALLREDUCE=ring
        export CCL_WORKER_COUNT=8

        for (( i = $SOCKETS; i < 2*$SOCKETS; i++ )); do  # pin CCL workers to HT
        START_CORE=$(( i * CORES ))
        for (( j = 0; j < $CCL_WORKER_COUNT; j++)); do
        CCL_WORKER_AFFINITY="${CCL_WORKER_AFFINITY} $((START_CORE + j))"
        done
        done

        export CCL_WORKER_AFFINITY=`echo ${CCL_WORKER_AFFINITY} | tr " " ","`

        #DDP settings
        export TORCH_CPP_LOG_LEVEL=INFO
        export TORCH_DISTRIBUTED_DEBUG=INFO
        export MASTER_ADDR=`head -1 hostfile`

        # Fabric settings
        export FI_PROVIDER=psm3
        export PSM3_IDENTIFY=1
        export PSM3_ALLOW_ROUTERS=1
        export PSM3_RDMA=1
        export PSM3_PRINT_STATS=0
        export PSM3_RV_MR_CACHE_SIZE=8192
        export PSM3_KASSIST_MODE=none
        #export PSM3_NIC='irdma*
        export FI_PSM3_CONN_TIMEOUT=100
        # export PSM3_HAL=sockets

        rm -rf ${OUTPUT_DIR}/LCM_${PRECISION}_dist_inference_accuracy*

        oneccl_bindings_for_pytorch_path=$(python -c "import torch; import oneccl_bindings_for_pytorch; import os;  print(os.path.abspath(os.path.dirname(oneccl_bindings_for_pytorch.__file__)))")
        source $oneccl_bindings_for_pytorch_path/env/setvars.sh
    fi
fi

if [[ "$TEST_MODE" == "ACCURACY" && "${DISTRIBUTED}" == "true" ]]; then
    python -m intel_extension_for_pytorch.cpu.launch \
        --nnodes ${NNODES} \
        --hostfile ${HOSTFILE} \
        --logical-cores-for-ccl --ccl-worker-count 8 \
        ${MODEL_DIR}/inference.py \
        --model_name_or_path="SimianLuo/LCM_Dreamshaper_v7" \
        --dataset_path=${DATASET_DIR} \
        --dist-backend ccl \
        --accuracy \
        $ARGS 2>&1 | tee ${OUTPUT_DIR}/LCM_${PRECISION}_dist_inference_accuracy.log

    # For the summary of results
    wait

elif [[ "${TEST_MODE}" == "ACCURACY" && "${DISTRIBUTED}" == "false" ]]; then
    if [[ "0" == ${TORCH_INDUCTOR} ]];then
        python -m intel_extension_for_pytorch.cpu.launch \
            --log-dir ${OUTPUT_DIR} \
            --log_file_prefix LCM_${PRECISION}_inference_${mode} \
            ${MODEL_DIR}/inference.py \
            --model_name_or_path="SimianLuo/LCM_Dreamshaper_v7" \
            --dataset_path=${DATASET_DIR} \
            --accuracy \
            $ARGS
    else
        python -m torch.backends.xeon.run_cpu --disable-numactl \
            --log_path ${OUTPUT_DIR} \
            ${MODEL_DIR}/inference.py \
            --model_name_or_path="SimianLuo/LCM_Dreamshaper_v7" \
            --dataset_path=${DATASET_DIR} \
            --accuracy \
            $ARGS 2>&1 | tee ${OUTPUT_DIR}/LCM_${PRECISION}_inference_throughput.log
    fi
    # For the summary of results
    wait
else
    if [[ "0" == ${TORCH_INDUCTOR} ]];then
        python -m intel_extension_for_pytorch.cpu.launch \
            --memory-allocator tcmalloc \
            $MODE_ARGS \
            --log-dir ${OUTPUT_DIR} \
            --log_file_prefix LCM_${PRECISION}_inference_${mode} \
            ${MODEL_DIR}/inference.py \
            --model_name_or_path="SimianLuo/LCM_Dreamshaper_v7" \
            --dataset_path=${DATASET_DIR} \
            --benchmark \
            -w ${num_warmup} -i ${num_iter} \
            $ARGS
    else
        python -m torch.backends.xeon.run_cpu --disable-numactl \
            --enable_tcmalloc \
            $MODE_ARGS \
            --log_path ${OUTPUT_DIR} \
            ${MODEL_DIR}/inference.py \
            --model_name_or_path="SimianLuo/LCM_Dreamshaper_v7" \
            --dataset_path=${DATASET_DIR} \
            --benchmark \
            -w ${num_warmup} -i ${num_iter} \
            $ARGS 2>&1 | tee ${OUTPUT_DIR}/LCM_${PRECISION}_inference_throughput.log
    fi
    # For the summary of results
    wait
fi

throughput="N/A"
accuracy="N/A"
latency="N/A"

if [[ "$TEST_MODE" == "REALTIME" ]]; then
    TOTAL_CORES=`expr $CORES \* $SOCKETS`
    INSTANCES=`expr $TOTAL_CORES / $CORES_PER_INSTANCE`
    INSTANCES_PER_SOCKET=`expr $INSTANCES / $SOCKETS`

    throughput=$(grep 'Throughput:' ${OUTPUT_DIR}/LCM_${PRECISION}_inference_latency* |sed -e 's/.*Throughput//;s/[^0-9.]//g' |awk -v INSTANCES_PER_SOCKET=$INSTANCES_PER_SOCKET '
    BEGIN {
            sum = 0;
            i = 0;
        }
        {
            sum = sum + $1;
            i++;
        }
    END   {
            sum = sum / i * INSTANCES_PER_SOCKET;
            printf("%.4f", sum);
    }')
    latency=$(grep 'Latency:' ${OUTPUT_DIR}/LCM_${PRECISION}_inference_latency* |sed -e 's/.*Latency//;s/[^0-9.]//g' |awk -v INSTANCES_PER_SOCKET=$INSTANCES_PER_SOCKET '
    BEGIN {
            sum = 0;
            i = 0;
        }
        {
            sum = sum + $1;
            i++;
        }
    END   {
            sum = sum / i * INSTANCES_PER_SOCKET;
            printf("%.4f", sum);
    }')
    echo "--------------------------------Performance Summary per Socket--------------------------------"
    echo ""LCM";"throughput";${PRECISION};${throughput}" | tee -a ${OUTPUT_DIR}/summary.log
    echo ""LCM";"latency";${PRECISION};${latency}" | tee -a ${OUTPUT_DIR}/summary.log

elif [[ "$TEST_MODE" == "THROUGHPUT" ]]; then
    throughput=$(grep 'Throughput:' ${OUTPUT_DIR}/LCM_${PRECISION}_inference_throughput* |sed -e 's/.*Throughput//;s/[^0-9.]//g' |awk '
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
            printf("%.4f", sum);
    }')
    latency=$(grep 'Latency:' ${OUTPUT_DIR}/LCM_${PRECISION}_inference_throughput* |sed -e 's/.*Latency//;s/[^0-9.]//g' |awk '
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
            printf("%.4f", sum);
    }')

    echo "--------------------------------Performance Summary per NUMA Node--------------------------------"
    echo ""LCM";"throughput";${PRECISION};${throughput}" | tee -a ${OUTPUT_DIR}/summary.log
    echo ""LCM";"latency";${PRECISION};${latency}" | tee -a ${OUTPUT_DIR}/summary.log
fi
if [[ "$TEST_MODE" == "ACCURACY" ]]; then
    if [[ "${DISTRIBUTED}" == "false" ]]; then
        accuracy=$(grep 'FID:' ${OUTPUT_DIR}/LCM_${PRECISION}_inference_accuracy* |sed -e 's/.*FID//;s/[^0-9.]//g')
        echo ""LCM";"FID";${PRECISION};${accuracy}" | tee -a ${OUTPUT_DIR}/summary.log
    elif [[ "${DISTRIBUTED}" == "true" ]]; then
        accuracy=$(grep 'FID:' ${OUTPUT_DIR}/LCM_${PRECISION}_dist_inference_accuracy* |sed -e 's/.*FID//;s/[^0-9.]//g')
        echo ""LCM";"FID";$1;${accuracy}" | tee -a ${OUTPUT_DIR}/summary.log
    fi
fi

yaml_content=$(cat << EOF
results:
- key : throughput
  value: $throughput
  unit: samples/sec
- key: latency
  value: $latency
  unit: s
- key: accuracy
  value: $accuracy
  unit: AP
EOF
)

echo "$yaml_content" >  $OUTPUT_DIR/results.yaml
echo "YAML file created."
