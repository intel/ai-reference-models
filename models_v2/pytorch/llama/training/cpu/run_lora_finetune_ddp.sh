
#!/bin/bash

#
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


ARGS=""

MAXSTEP=${MAXSTEP:-50}

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
#export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set, please create the output path and set it to OUTPUT_DIR"
  exit 1
fi

if [[ "$1" == "bf16" ]]
then
    precision="bf16"
    ARGS="$ARGS --bf16 "
    echo "### running bf16 mode"
elif [[ "$1" == "fp32" ]]
then
    echo "### running fp32 mode"
elif [[ "$1" == "fp16" ]]
then
    precision=fp16
    ARGS="$ARGS --fp16 "
    echo "### running fp16 mode"
elif [[ "$1" == "bf32" ]]
then
    precision=bf32
    ARGS="$ARGS --bf32 "
    echo "### running bf32 mode"
else
    echo "The specified precision '$1' is unsupported."
    echo "Supported precisions are: fp32, bf32, bf16, fp16"
    exit 1
fi

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`
NNODES=${NNODES:-1}
HOSTFILE=${HOSTFILE:-./hostfile}
NUM_RANKS=$(( NNODES * SOCKETS ))

CORES_PER_INSTANCE=$CORES

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export KMP_BLOCKTIME=1
export KMP_AFFINITY=granularity=fine,compact,1,0

<< EOF
#oneCCL settings
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
EOF

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
export PSM3_HAL=sockets


oneccl_bindings_for_pytorch_path=$(python -c "import torch; import oneccl_bindings_for_pytorch; import os;  print(os.path.abspath(os.path.dirname(oneccl_bindings_for_pytorch.__file__)))")
source $oneccl_bindings_for_pytorch_path/env/setvars.sh

#export FI_PROVIDER_PATH=$oneccl_bindings_for_pytorch_path/lib/prov
python -m intel_extension_for_pytorch.cpu.launch \
    --memory-allocator jemalloc \
    --nnodes ${NNODES} \
    --hostfile ${HOSTFILE} \
    --logical-cores-for-ccl --ccl_worker_count 8 \
    ../../../../../../models/language_modeling/pytorch/llama/training/cpu/finetune.py  $ARGS \
    --base_model 'meta-llama/Llama-2-7b-hf'\
    --data_path '../../../../../../models/language_modeling/pytorch/llama/training/cpu/alpaca_data.json' \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 32 \
    --micro_batch_size 32 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --val_set_size 2000 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --group_by_length \
    --max_steps ${MAXSTEP} 


