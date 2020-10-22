#!/usr/bin/env bash
#
# Copyright (c) 2020 Intel Corporation
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

FROZEN_GRAPH=${FROZEN_GRAPH-"$MODEL_DIR/transformer_lt_official_fp32_pretrained_model/graph/fp32_graphdef.pb"}
if [[ ! -f "${FROZEN_GRAPH}" ]]; then
  # If the frozen graph is not found, check if we have to untar the file
  tar -xvf transformer_lt_official_fp32_pretrained_model.tar.gz

  if [[ ! -f "${FROZEN_GRAPH}" ]]; then
    echo "The frozen graph could not be found at $FROZEN_GRAPH"
    exit 1
  fi
fi

# vars for english, german, and vocab file names
EN_DATA_FILE=${EN_DATA_FILE-newstest2014.en}
DE_DATA_FILE=${DE_DATA_FILE-newstest2014.de}
VOCAB_FILE=${VOCAB_FILE-vocab.txt}

source "$(dirname $0)/common/utils.sh"
_command python ${MODEL_DIR}/benchmarks/launch_benchmark.py \
  --model-name transformer_lt_official \
  --precision fp32 \
  --mode inference \
  --framework tensorflow \
  --batch-size 64 \
  --socket-id 0 \
  --in-graph ${FROZEN_GRAPH} \
  --data-location ${DATASET_DIR} \
  --output-dir ${OUTPUT_DIR} \
  $@ \
  -- file=${EN_DATA_FILE} \
  file_out=translate.txt \
  reference=${DE_DATA_FILE} \
  vocab_file=${VOCAB_FILE}
