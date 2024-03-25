#!/usr/bin/env bash
#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
#
# AGPL-3.0 license
#


MODEL_DIR=${MODEL_DIR-$PWD}

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

if [ -z "${DATASET_DIR}" ]; then
  echo "The required environment variable DATASET_DIR has not been set"
  exit 1
fi

if [ ! -d "${DATASET_DIR}" ]; then
  echo "The DATASET_DIR '${DATASET_DIR}' does not exist"
  exit 1
fi

if [ -z "${PRECISION}" ]; then
  echo "The required environment variable PRECISION has not been set"
  echo "Please set PRECISION to fp32, bfloat16, fp16, int8"
  exit 1
fi

if [[ $PRECISION != "fp32" ]] && [[ $PRECISION != "int8" ]] && [[ $PRECISION != "bfloat16" ]] && [[ $PRECISION != "fp16" ]]; then
  echo "The specified precision '${PRECISION}' is unsupported."
  echo "Supported precisions are: fp32, bfloat16, fp16 and int8"
  exit 1
fi

if [[ ! -f "${PRETRAINED_MODEL}" ]]; then
  echo "The file specified by the PRETRAINED_MODEL environment variable (${PRETRAINED_MODEL}) does not exist."
  exit 1
fi

# If batch size env is not mentioned, then the workload will run with the default batch size.
if [ -z "${BATCH_SIZE}" ]; then
  BATCH_SIZE="1"
  echo "Running with default batch size of ${BATCH_SIZE}"
fi

if [[ $PRECISION == "fp16" ]]; then
  export ONEDNN_MAX_CPU_ISA=AVX512_CORE_AMX_FP16
  echo "ONEDNN_MAX_CPU_ISA=$ONEDNN_MAX_CPU_ISA"
fi

source "${MODEL_DIR}/quickstart/common/utils.sh"
_command python benchmarks/launch_benchmark.py \
         --model-name=yolov5 \
         --precision ${PRECISION} \
         --mode=inference \
         --framework tensorflow \
         --in-graph ${PRETRAINED_MODEL} \
         --data-location=${DATASET_DIR} \
         --output-dir ${OUTPUT_DIR} \
         --batch-size=${BATCH_SIZE} \
         --socket-id 0 \
         --benchmark-only \
         $@ \
         $ARGS
