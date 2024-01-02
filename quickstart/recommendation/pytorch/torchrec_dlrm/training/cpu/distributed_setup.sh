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

oneccl_bindings_for_pytorch_path=$(python -c "import torch; import oneccl_bindings_for_pytorch; import os;  print(os.path.abspath(os.path.dirname(oneccl_bindings_for_pytorch.__file__)))")
source $oneccl_bindings_for_pytorch_path/env/setvars.sh
export NODE=${NODE:-1}
export NUM_CCL_WORKER=${NUM_CCL_WORKER:-1}
export NUM_CCL_WORKER=${NUM_CCL_WORKER:-1}
export HOSTFILE=${HOSTFILE:-hostfile1}
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO
export CCL_LOG_LEVEL=info
export CCL_ALLREDUCE=rabenseifner
export LOGICAL_CORE_CCL=${LOGICAL_CORE_CCL:-0}
CORES_PER_SOCKET=`lscpu | grep "Core(s) per socket" | awk '{print $4}'`
SOCKETS=`lscpu | grep "Socket(s)" | awk '{print $2}'`
NUMA_NODES=`lscpu | grep "NUMA node(s)" | awk '{print $3}'`
NUMA_NODES_PER_SOCKETS=`expr $NUMA_NODES / $SOCKETS`
CORES_PER_NUMA_NODE=`expr $CORES_PER_SOCKET / $NUMA_NODES_PER_SOCKETS`

NODE_LIST=${NODE_LIST:-"0"}
if [[ "0" == ${NODE_LIST} ]];then
    if [ $NUMA_NODES_PER_SOCKETS -eq 1 ]
    then 
        NODE_LIST="0,1"
        PROC_PER_NODE=${PROC_PER_NODE:-"2"}
    else
        for i in $(seq 1 `expr $NUMA_NODES_PER_SOCKETS - 1`)
        do
            NODE_LIST="${NODE_LIST},$i"
        done
        PROC_PER_NODE=${PROC_PER_NODE:-"$NUMA_NODES_PER_SOCKETS"}
    fi
fi

export launcher_dist_args="--nodes-list ${NODE_LIST} --nprocs-per-node=$PROC_PER_NODE --ccl_worker_count $NUM_CCL_WORKER --distributed --hostfile $HOSTFILE --nnodes $NODE "
if [[ "1" == ${NODE_LIST} ]];then
export launcher_dist_args = "${launcher_dist_args} --logical_core_for_ccl"
fi
export W_SIZE=`expr $PROC_PER_NODE \* $NODE`
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29550}

echo "NODE: $NODE"
echo "NODE_LIST: $NODE_LIST"
echo "PROC_PER_NODE: $PROC_PER_NODE"
echo "W_SIZE: $W_SIZE"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "NUM_CCL_WORKER: $NUM_CCL_WORKER"
echo "HOSTFILE: $HOSTFILE"
echo "launcher_dist_args: $launcher_dist_args"