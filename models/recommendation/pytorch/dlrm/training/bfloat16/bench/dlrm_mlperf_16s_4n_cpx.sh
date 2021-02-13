#!/bin/bash

seed_num=$(date +%s)
export KMP_BLOCKTIME=1
export KMP_AFFINITY="granularity=fine,compact,1,0"
export LD_PRELOAD="${CONDA_PREFIX}/lib/libtcmalloc.so:${CONDA_PREFIX}/lib/libiomp5.so"
export CCL_WORKER_COUNT=3
export CCL_WORKER_AFFINITY="0,1,2,28,29,30,56,57,58,84,85,86"
export CCL_ATL_TRANSPORT=ofi
export MASTER_ADDR=`mpiexec.hydra -np 1 -f hostfile hostname`

Config="mpiexec.hydra -np 16 -ppn 4 -f hostfile -l -genv I_MPI_PIN_DOMAIN=[0x0000000FFFFFF0,0xFFFFFF00000000,0x0000000FFFFFF000000000000000,0xFFFFFF0000000000000000000000] -genv OMP_NUM_THREADS=24"

$Config python -u dlrm_s_pytorch.py --arch-sparse-feature-size=128 --arch-mlp-bot="13-512-256-128" --arch-mlp-top="1024-1024-512-256-1" --max-ind-range=40000000  --data-generation=dataset --data-set=terabyte --raw-data-file=$DATA_PATH/day --processed-data-file=$DATA_PATH/terabyte_processed.npz --loss-function=bce --round-targets=True --num-workers=0 --test-num-workers=0 --use-ipex --bf16  --dist-backend=ccl --learning-rate=18.0 --mini-batch-size=32768 --print-freq=128 --print-time --test-freq=6400 --test-mini-batch-size=65536 --lr-num-warmup-steps=8000 --lr-decay-start-step=70000 --lr-num-decay-steps=30000 --memory-map --mlperf-logging --mlperf-auc-threshold=0.8025 --mlperf-bin-loader --mlperf-bin-shuffle --numpy-rand-seed=$seed_num $dlrm_extra_option 2>&1 | tee run_terabyte_mlperf_32k_16_sockets_${seed_num}.log
