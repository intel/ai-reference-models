#!/bin/bash
#pip install "git+https://github.com/mlperf/logging.git@
MODEL_DIR=${MODEL_DIR-../../../../../../}
SCRIPT=${SCRIPT:-${MODEL_DIR}/models/language_modeling/pytorch/fast_bert/pretrain_mlperf/run_pretrain_mlperf.py}
PRETRAINED_MODEL=${PRETRAINED_MODEL:-/scratch1/bert/bert/checkpoint}
DATASET_DIR=${DATASET_DIR:-/scratch1/bert/bert/bert_dataset/hdf5}
NUMA_ARGS=""
NUM_RANKS=1
echo "MPI_LOCALRANKID=$MPI_LOCALRANKID  PMI_SIZE=$PMI_SIZE"
if [ "x$MPI_LOCALRANKID" != "x" ] ; then
  REAL_NUM_NUMA_NODES=`lscpu | grep "NUMA node(s):" | awk '{print $NF}'`
  PPNUMA=$(( MPI_LOCALNRANKS / REAL_NUM_NUMA_NODES ))
  if [ $PPNUMA -eq 0 ] ; then 
    if [ "x$SINGLE_SOCKET_ONLY" == "x1" ] ; then 
      if command -v numactl >& /dev/null ; then
        NUMA_ARGS="numactl -m 0 "
      fi
    fi
  else
    NUMARANK=$(( MPI_LOCALRANKID / PPNUMA ))
    NUMA_ARGS="$NUMA_ARGS $GDB_ARGS "
  fi
  NUM_RANKS=$PMI_SIZE
  echo "setting NUM_RANKS=$NUM_RANKS"
else
  if command -v numactl >& /dev/null ; then
    NUMA_ARGS="numactl -m 0 "
  fi
  NUM_RANKS=1
fi

echo "NUM_RANKS=$NUM_RANKS"
# CLEAR YOUR CACHE HERE
python -c "
from mlperf_logging.mllog import constants
from mlperf_logger import log_event
log_event(key=constants.CACHE_CLEAR, value=True)"


GBS=2048
GBS=${GBS:-2048}
LBS=$(( GBS / NUM_RANKS ))

params="--train_batch_size=${LBS}    --learning_rate=2.0e-3     --opt_lamb_beta_1=0.66     --opt_lamb_beta_2=0.998     --warmup_proportion=0.0     --warmup_steps=0.0     --start_warmup_step=0     --max_steps=1710   --phase2    --max_predictions_per_seq=76      --do_train     --skip_checkpoint     --train_mlm_accuracy_window_size=0     --target_mlm_accuracy=0.720     --weight_decay_rate=0.01     --max_samples_termination=4500000     --eval_iter_start_samples=150000 --eval_iter_samples=150000     --eval_batch_size=16  --gradient_accumulation_steps=1     --log_freq=0 "

$NUMA_ARGS $GDB_ARGS python -u $SCRIPT \
    --input_dir ${DATASET_DIR}/training-4320/hdf5_4320_shards_varlength/ \
    --eval_dir ${DATASET_DIR}/eval_varlength/ \
    --model_type 'bert' \
    --model_name_or_path $PRETRAINED_MODEL \
    --output_dir model_save \
    $params \
    $@
