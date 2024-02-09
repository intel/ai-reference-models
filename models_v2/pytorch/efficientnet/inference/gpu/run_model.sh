# Copyright (c) 2023-2024 Intel Corporation
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

#!/bin/bash

# Specify default arguments

[[ "${AMP}" == "" ]]            && AMP="yes"
[[ "${BATCH_SIZE}" == "" ]]     && BATCH_SIZE=1
[[ "${DATASET_DIR}" == "" ]]    && DATASET_DIR=""
[[ "${DUMMY}" == "" ]]          && DUMMY="no"
[[ "${LOAD_PATH}" == "" ]]      && LOAD_PATH=""
[[ "${JIT}" == "" ]]            && JIT="trace"
[[ "${MODEL_NAME}" == "" ]]     && MODEL_NAME="efficientnet_b0"
[[ "${MULTI_TILE}" == "" ]]     && MULTI_TILE="False"
[[ "${NUM_IMAGES}" == "" ]]     && NUM_IMAGES=1
[[ "${NUM_ITERATIONS}" == "" ]] && NUM_ITERATIONS=100
[[ "${PRECISION}" == "" ]]      && PRECISION="fp32"
[[ "${SAVE_PATH}" == "" ]]      && SAVE_PATH=""
[[ "${STATUS_PRINTS}" == "" ]]  && STATUS_PRINTS=10
[[ "${STREAMS}" == "" ]]        && STREAMS=1

# Process CLI arguments as overides for environment variables
VALID_ARGS=$(getopt -o h --long amp:,arch:,batch-size:,data:,dummy,help,load:,jit:,multi-tile,num-images:,num-iterations:,output-dir:,platform:,precision:,proxy:,save:,status-prints:,streams: -- "$@")
if [[ $? -ne 0 ]]; then
    exit 1;
fi
eval set -- "$VALID_ARGS"
while [ : ]; do
  case "$1" in
    --amp)
        AMP="$2"
        shift 2
        ;;
    --arch)
        MODEL_NAME=$2
        shift 2
        ;;
    --batch-size)
        BATCH_SIZE=$2
        shift 2
        ;;
    --data)
        DATASET_DIR="$2"
        shift 2
        ;;
    --dummy)
        DUMMY="yes"
        shift 1
        ;;
    --jit)
        JIT="$2"
        shift 2
        ;;
    --load)
        LOAD_PATH="$2"
        shift 2
        ;;
    --multi-tile)
        MULTI_TILE="True"
        shift 1
        ;;
    --num-images)
        NUM_IMAGES=$2
        shift 2
        ;;
    --num-iterations)
        NUM_ITERATIONS=$2
        shift 2
        ;;
    --output-dir)
        OUTPUT_DIR=$2
        shift 2
        ;;
    --platform)
        PLATFORM=$2
        shift 2
        ;;
    --precision)
        PRECISION=$2
        shift 2
        ;;
    --proxy)
        PROXY=$2
        shift 2
        ;;
    --save)
        SAVE_PATH="$2"
        shift 2
        ;;
    --status-prints)
        STATUS_PRINTS=$2
        shift 2
        ;;
    --streams)
        STREAMS=$2
        shift 2
        ;;
    -h | --help)
        echo "Usage: $(basename $0)"
        echo "  --amp            [AMP]           : Use AMP on model conversion (default: '${AMP}')"
        echo "                                     * no"
        echo "                                     * yes"
        echo "  --arch           [MODEL_NAME]    : Specify torchvision model to run (default: ${MODEL_NAME}):"
        echo "                                     * efficientnet_b0"
        echo "                                     * efficientnet_b1"
        echo "                                     * efficientnet_b2"
        echo "                                     * efficientnet_b3"
        echo "                                     * efficientnet_b4"
        echo "                                     * efficientnet_b5"
        echo "                                     * efficientnet_b6"
        echo "                                     * efficientnet_b7"
        echo "  --batch-size     [BATCH_SIZE]    : Batch size to use (default: '${BATCH_SIZE}')"
        echo "  --data           [DATASET_DIR]   : Location to load images from (default: '${DATASET_DIR}')"
        echo "  --dummy                          : Use randomly generated dummy dataset in place of '--data' argument (default: disabled)"
        echo "  --load           [LOAD_PATH]     : If specified model will be loaded from this saved location (default: disabled)"
        echo "  --jit            [JIT]           : JIT method to use (default: '${JIT}')"
        echo "                                     * none"
        echo "                                     * script"
        echo "                                     * trace"
        echo "  --multi-tile                     : Run benchmark in multi-tile configuration (default: '${MULTI_TILE}')"
        echo "  --num-images     [NUM_IMAGES]    : Number of images to load (default: '${NUM_IMAGES}')"
        echo "  --num-iterations [NUM_ITERATIONS]: Number of times to test each batch (default: '${NUM_ITERATIONS}')"
        echo "  --output-dir     [OUTPUT_DIR]    : Location to write output to. Required"
        echo "  --platform       [PLATFORM]      : Platform that inference is being ran on (default: '${PLATFORM}')"
        echo "                                     * CPU"
        echo "                                     * ATS-M"
        echo "                                     * CUDA"
        echo "                                     * PVC"
        echo "  --precision      [PRECISION]     : Precision to use for the model (default: '${PRECISION}')"
        echo "                                     * bf16"
        echo "                                     * fp16"
        echo "                                     * fp32"
        echo "                                     * int8"
        echo "  --proxy          [PROXY]         : System proxy. Required to download models"
        echo "  --save           [SAVE_PATH]     : If specified model will be saved to this saved location (default: disabled)"
        echo "  --status-prints  [STATUS_PRINTS] : Total number of status messages to display during inference benchmarking (default: '${STATUS_PRINTS}')"
        echo "  --streams        [STREAMS]       : Number of parallel streams to do inference on (default: '${STREAMS}')"
        echo "                                     Will be truncated to a multiple of BATCH_SIZE"
        echo "                                     If less than BATCH_SIZE will be increased to BATCH_SIZE"
        echo ""
        echo "NOTE: Arguments may also be specified through command line variables using the name in '[]'."
        echo "      For example 'export MODEL_NAME=efficientnet_b0'."
        echo "NOTE: Both arguments and their values are case sensitive."
        exit 0
        ;;
    --) shift;
        break
        ;;
  esac
done

# Check data set is configured if specified/required.
if [[ ${DUMMY} == "yes" ]]; then
    if [[ -z "${DATASET_DIR}" ]]; then
        # Dummy data requested and no dataset provided. OK case.
        _dataset_args="--dummy"
    else
        # Ambiguous case. Terminate.
        echo "ERROR: Cannot handle simultaneous usage of '--dummy'/'DUMMY' and '--data'/'DATASET_DIR'."
        exit 1
    fi
else
    if [[ -z "${DATASET_DIR}" ]]; then
        # Ambiguous case. Terminate.
        echo "ERROR: Dataset must be provided or dummy data must be enabled:"
        echo "ERROR: add '--dummy' or '--data [DATASET_DIR]'."
        exit 1
    else
        if [[ -d "${DATASET_DIR}" ]]; then
            # Actual dataset provided provided. OK case.
            _dataset_args="--data ${DATASET_DIR}"
        else
            echo "ERROR: The requested dataset '${DATASET_DIR}' does not exist!"
            exit 1
        fi
    fi
fi

# Check multi-tile is only specified on valid platforms.
if [[ "${MULTI_TILE}" == "True" ]]; then
    if [[ "${PLATFORM}" == "PVC" ]]; then
        echo "Streams will be round-robin scheduled across multiple tiles"
        if [ $((STREAMS%2)) -ne 0 ]; then
        echo "WARNING: can't schedule evenly odd number of streams ($STREAMS) across tiles"
    fi
    fi
    if [[ "${PLATFORM}" == "ATS-M" ]]; then
        echo "ERROR: ATS-M does not support multitile"
        exit 1
    fi
    if [[ "${PLATFORM}" == "CUDA" ]]; then
        echo "ERROR: multitile is not implemented for CUDA"
        exit 1
    fi
fi

# Show test configuration
echo 'Running with parameters:'
echo " AMP:            ${AMP}"
echo " BATCH_SIZE:     ${BATCH_SIZE}"
echo " DATASET_DIR:    ${DATASET_DIR}"
echo " DUMMY:          ${DUMMY}"
echo " JIT:            ${JIT}"
echo " LOAD_PATH:      ${LOAD_PATH}"
echo " MODEL_NAME:     ${MODEL_NAME}"
echo " MULTI_TILE:     ${MULTI_TILE}"
echo " NUM_ITERATIONS: ${NUM_ITERATIONS}"
echo " NUM_IMAGES:     ${NUM_IMAGES}"
echo " OUTPUT_DIR:     ${OUTPUT_DIR}"
echo " SAVE_PATH:      ${SAVE_PATH}"
echo " STATUS_PRINTS:  ${STATUS_PRINTS}"
echo " STREAMS:        ${STREAMS}"
echo " PLATFORM:       ${PLATFORM}"
echo " PRECISION:      ${PRECISION}"
echo " PROXY:          ${PROXY}"

# Set system proxies if requested.
if [[ "${PROXY}" != "" ]]; then
    export http_proxy=${PROXY}
    export https_proxy=${PROXY}
fi

# known issue for multitile
if [[ "${MULTI_TILE}" == "True" ]]; then
    export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
fi

# Configure save and load params
_save_load_args=""
if [[ ${LOAD_PATH} != "" ]]; then
    _save_load_args="${_save_load_args} --load ${LOAD_PATH}"
fi
if [[ ${SAVE_PATH} != "" ]]; then
    _save_load_args="${_save_load_args} --save ${SAVE_PATH}"
fi

# Specify image dimensions for model
if [[ ${MODEL_NAME} == "efficientnet_b0" ]]; then
    _img_width=224
    _img_height=224
elif [[ ${MODEL_NAME} == "efficientnet_b1" ]]; then
    _img_width=240
    _img_height=240
elif [[ ${MODEL_NAME} == "efficientnet_b2" ]]; then
    _img_width=260
    _img_height=260
elif [[ ${MODEL_NAME} == "efficientnet_b3" ]]; then
    _img_width=300
    _img_height=300
elif [[ ${MODEL_NAME} == "efficientnet_b4" ]]; then
    _img_width=380
    _img_height=380
elif [[ ${MODEL_NAME} == "efficientnet_b5" ]]; then
    _img_width=456
    _img_height=456
elif [[ ${MODEL_NAME} == "efficientnet_b6" ]]; then
    _img_width=528
    _img_height=528
elif [[ ${MODEL_NAME} == "efficientnet_b7" ]]; then
    _img_width=600
    _img_height=600
else
    echo "ERROR: Model architecture '${MODEL_NAME}' is not supported yet"
    exit 1
fi

# Specify data type
if [[ ${PRECISION} == "fp32" ]]; then
    _dtype_args=""
elif [[ ${PRECISION} == "fp16" ]]; then
    _dtype_args="--fp16 1"
elif [[ ${PRECISION} == "bf16" ]]; then
    _dtype_args="--bf16 1"
elif [[ ${PRECISION} == "int8" ]]; then
    #_dtype_args="--int8 1 --asymmetric-quantization --perchannel-weight 1"
    echo "ERROR: Precision '${PRECISION}' is not supported yet for model '${MODEL_NAME}'"
    exit 1
fi

# Specify if AMP should be used
if [[ ${AMP} == "no" ]]; then
    _amp_arg="--no-amp"
elif [[ ${AMP} == "yes" ]]; then
    _amp_arg=""
else
    echo "ERROR: Invalid valid entered for 'AMP': ${AMP}"
    exit 1
fi

# Specify if JIT should be used
if [[ ${JIT} == "none" ]]; then
    _jit_arg=""
elif [[ ${JIT} == "trace" ]]; then # Only specifiable through environment variables.
    _jit_arg="--jit-trace"
elif [[ ${JIT} == "script" ]]; then # Only specifiable through environment variables.
    _jit_arg="--jit-script"
else
    echo "ERROR: Invalid valid entered for 'JIT': ${JIT}"
    exit 1
fi

# General perf args
_perf_args="--pretrained --channels-last --no-grad"

# Create the output directory, if it doesn't already exist
mkdir -p $OUTPUT_DIR

# Set environment variables
if [[ ${PLATFORM} == "ATS-M" ]]; then
    export IGC_EnableDPEmulation=1
    export CFESingleSliceDispatchCCSMode=1
    export IPEX_ONEDNN_LAYOUT=1
    export IPEX_LAYOUT_OPT=1
elif [[ ${PLATFORM} == "PVC" ]]; then
    # Currently its an assumption that PCV uses these.
    export IGC_EnableDPEmulation=1
    export CFESingleSliceDispatchCCSMode=1
    export IPEX_ONEDNN_LAYOUT=1
    export IPEX_LAYOUT_OPT=1
fi
export PROFILE="OFF"

# Start inference script with numactl
echo "Starting inference..."
#TODO: Set ZE_AFFINITY_MASK for multiple tiles.
numactl --cpunodebind=0 --membind=0 python3 predict.py \
    --arch ${MODEL_NAME} \
    ${_dataset_args} \
    --batch-size ${BATCH_SIZE} \
    --status-prints ${STATUS_PRINTS} \
    --max-val-dataset-size ${NUM_IMAGES} \
    --batch-streaming ${NUM_ITERATIONS} \
    --width ${_img_width} --height ${_img_height} \
    ${_dtype_args} ${_amp_arg} ${_jit_arg} ${_perf_args} ${_save_load_args} \
    --warm-up 10 \
    --output-dir ${OUTPUT_DIR} \
    --total-instances ${STREAMS} \
    --terminate-if-sync-fail \
    2>&1 | tee ${OUTPUT_DIR}/output_raw.log
predict_exit_code=${PIPESTATUS[0]}
echo "Inference complete"

echo "Output directory contents:"
ls ${OUTPUT_DIR}

if [ $predict_exit_code -ne 0 ]; then
    echo "ERROR: Model scripts terminated with non-zero exit code: $predict_exit_code"
    exit 1
fi
exit 0
