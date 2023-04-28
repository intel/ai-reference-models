#!/bin/bash
MODEL_DIR=${MODEL_DIR-../../../../../../}
SCRIPT=${SCRIPT:-${MODEL_DIR}/models/language_modeling/pytorch/fast_bert/squad/run_squad.py}
EVAL_DATA_FILE=${EVAL_DATA_FILE:-"${PWD}/squad1.1/dev-v1.1.json"}
FINTUNE_DATA_FILE=${FINTUNE_DATA_FILE:-"${PWD}/squad1.1/train-v1.1.json"}
FINETUNED_MODEL=${FINETUNED_MODEL:-bert_squad_model}
NUMA_ARGS=""
if command -v numactl >& /dev/null ; then
if [ "x$MPI_LOCALRANKID" != "x" ] ; then
  REAL_NUM_NUMA_NODES=`lscpu | grep "NUMA node(s):" | awk '{print $NF}'`
  PPNUMA=$(( MPI_LOCALNRANKS / REAL_NUM_NUMA_NODES ))
  if [ $PPNUMA -eq 0 ] ; then 
    if [ "x$SINGLE_SOCKET_ONLY" == "x1" ] ; then 
      NUMA_ARGS="numactl -m 0 "
    fi
  else
    NUMARANK=$(( MPI_LOCALRANKID / PPNUMA ))
    NUMA_ARGS="$NUMA_ARGS $GDB_ARGS "
  fi
  NUM_RANKS=$PMI_SIZE
else
  NUMA_ARGS="numactl -m 0 "
  NUM_RANKS=1
fi
fi

$NUMA_ARGS $GDB_ARGS python -u ${SCRIPT} \
  --model_type bert \
  --model_name_or_path bert-large-uncased-whole-word-masking \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file ${FINTUNE_DATA_FILE} \
  --predict_file ${EVAL_DATA_FILE} \
  --per_gpu_train_batch_size 24 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /tmp/debug_squad/ \
  $@

