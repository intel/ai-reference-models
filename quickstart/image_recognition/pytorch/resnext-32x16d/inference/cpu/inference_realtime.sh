#!/usr/bin/env bash
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

MODEL_DIR=${MODEL_DIR-$PWD}

echo "PRECISION: $PRECISION"
echo "OUTPUT_DIR: $OUTPUT_DIR"

if [ -z "${OUTPUT_DIR}" ]; then
  echo "The required environment variable OUTPUT_DIR has not been set"
  exit 1
fi

# Create the output directory in case it doesn't already exist
mkdir -p ${OUTPUT_DIR}

if [ -z "${PRECISION}" ]; then
  echo "The required environment variable PRECISION has not been set"
  echo "Please set PRECISION to fp32, int8, avx-int8, or bf16."
  exit 1
fi

cd ${MODEL_DIR}/models/resnext-32x16d/examples/imagenet

# download pretrained weight.
python hub_help.py --url https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnext101_32x16-f3559a9c.pth

export work_space=${OUTPUT_DIR}

if [[ $PRECISION == "int8" || $PRECISION == "avx-int8" ]]; then
    if [[ $PRECISION == "avx-int8" ]]; then
        unset DNNL_MAX_CPU_ISA
    fi
    bash run_multi_instance_latency_ipex_spr.sh resnext101_32x16d_swsl int8 no_jit resnext101_configure_sym.json 2>&1 | tee -a ${OUTPUT_DIR}/resnext-32x16d-inference-realtime-int8.log
elif [[ $PRECISION == "bf16" ]]; then
    bash run_multi_instance_latency_ipex_spr.sh resnext101_32x16d_swsl bf16 jit 2>&1 | tee -a ${OUTPUT_DIR}/resnext-32x16d-inference-realtime-bf16.log
elif [[ $PRECISION == "fp32" ]]; then
    bash run_multi_instance_latency_ipex_spr.sh resnext101_32x16d_swsl fp32 jit 2>&1 | tee -a ${OUTPUT_DIR}/resnext-32x16d-inference-realtime-fp32.log
else
    echo "The specified precision '${PRECISION}' is unsupported."
    echo "Supported precisions are: fp32, bf16, int8, and avx-int8"
    exit 1
fi
