#!/bin/sh

###############################################################################
### How to run?
### Test cpu accuracy. Just run
###
###############################################################################
export KMP_AFFINITY="granularity=fine,compact,1,0"
export KMP_BLOCKTIME=1
export LD_PRELOAD="${CONDA_PREFIX}/lib/libiomp5.so:$LDPRELOAD"
export DNNL_PRIMITIVE_CACHE_CAPACITY=1024
export KMP_SETTINGS=1

ARGS=""

if [[ "$1" == "dnnl" ]]
then
    ARGS="$ARGS --ipex --dnnl"
    echo "### running auto_dnnl mode"
fi

if [[ "$2" == "int8" ]]
then
    ARGS="$ARGS --int8 --int8-calibration"
    echo "### running int8 mode"
fi


CORES=`lscpu | grep Core | awk '{print $4}'`
# use first socket
numa_cmd="numactl -C 0-$((CORES-1))  -m 0"
echo "will run on core 0-$((CORES-1)) on socket 0"

export OMP_NUM_THREADS=1
$numa_cmd python -u dlrm_s_pytorch.py \
--raw-data-file=${DATASET_PATH}/day --processed-data-file=${DATASET_PATH}/terabyte_processed.npz \
--loss-function=bce --data-generation=dataset --data-set=terabyte \
--memory-map --mlperf-bin-loader --round-targets=True --learning-rate=1.0 \
--arch-mlp-bot=13-512-256-128 --arch-mlp-top=1024-1024-512-256-1 \
--arch-sparse-feature-size=128 --max-ind-range=40000000 \
--numpy-rand-seed=727  --inference-only --jit \
--print-freq=100 --print-time --mini-batch-size=16 \
--share-weight --num-instance=$CORES $ARGS
