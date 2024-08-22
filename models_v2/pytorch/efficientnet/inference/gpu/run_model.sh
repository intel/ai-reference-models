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

[[ "${AMP}" == "" ]]               && AMP="yes"
[[ "${BATCH_SIZE}" == "" ]]        && BATCH_SIZE=1
[[ "${DATASET_DIR}" == "" ]]       && DATASET_DIR=""
[[ "${DUMMY}" == "" ]]             && DUMMY="no"
[[ "${JIT}" == "" ]]               && JIT="trace"
[[ "${LOAD_PATH}" == "" ]]         && LOAD_PATH=""
[[ "${MAX_TEST_DURATION}" == "" ]] && MAX_TEST_DURATION=""
[[ "${MIN_TEST_DURATION}" == "" ]] && MIN_TEST_DURATION=""
[[ "${MODEL_NAME}" == "" ]]        && MODEL_NAME="efficientnet_b0"
[[ "${MULTI_TILE}" == "" ]]        && MULTI_TILE="False"
[[ "${NUM_INPUTS}" == "" ]]        && NUM_INPUTS=1
[[ "${PRECISION}" == "" ]]         && PRECISION="fp32"
[[ "${PRINT_FREQUENCY}" == "" ]]   && PRINT_FREQUENCY=2
[[ "${SAVE_PATH}" == "" ]]         && SAVE_PATH=""
[[ "${SOCKET}" == "" ]]            && SOCKET=""
[[ "${STREAMS}" == "" ]]           && STREAMS=1
[[ "${IPEX}" == "" ]]              && IPEX="yes"

# Process CLI arguments as overides for environment variables
VALID_ARGS=$(getopt -o h --long amp:,arch:,batch-size:,data:,dummy,help,load:,jit:,max-test-duration:,min-test-duration:,multi-tile,num-inputs:,output-dir:,platform:,precision:,print-frequency:,proxy:,save:,socket:,streams:,ipex: -- "$@")
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
    --max-test-duration)
        MAX_TEST_DURATION="$2"
        shift 2
        ;;
    --min-test-duration)
        MIN_TEST_DURATION="$2"
        shift 2
        ;;
    --multi-tile)
        MULTI_TILE="True"
        shift 1
        ;;
    --num-inputs)
        NUM_INPUTS=$2
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
    --ipex)
        IPEX=$2
        shift 2
        ;;
    --precision)
        PRECISION=$2
        shift 2
        ;;
    --print-frequency)
        PRINT_FREQUENCY=$2
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
    --socket)
        SOCKET="$2"
        shift 2
        ;;
    --streams)
        STREAMS=$2
        shift 2
        ;;
    -h | --help)
        echo "Usage: $(basename $0)"
        echo "  --amp               [AMP]              : Use AMP on model conversion (default: '${AMP}')"
        echo "                                           * no"
        echo "                                           * yes"
        echo "  --arch              [MODEL_NAME]       : Specify torchvision model to run (default: ${MODEL_NAME}):"
        echo "                                           * efficientnet_b0"
        echo "                                           * efficientnet_b1"
        echo "                                           * efficientnet_b2"
        echo "                                           * efficientnet_b3"
        echo "                                           * efficientnet_b4"
        echo "                                           * efficientnet_b5"
        echo "                                           * efficientnet_b6"
        echo "                                           * efficientnet_b7"
        echo "  --batch-size        [BATCH_SIZE]       : Batch size to use (default: '${BATCH_SIZE}')"
        echo "  --data              [DATASET_DIR]      : Location to load images from (default: '${DATASET_DIR}')"
        echo "  --dummy                                : Use randomly generated dummy dataset in place of '--data' argument (default: disabled)"
        echo "  --load              [LOAD_PATH]        : If specified model will be loaded from this saved location (default: disabled)"
        echo "  --jit               [JIT]              : JIT method to use (default: '${JIT}')"
        echo "                                           * none"
        echo "                                           * compile"
        echo "                                           * script"
        echo "                                           * trace"
        echo "  --max-test-duration [MAX_TEST_DURATION]: Maximum duration in seconds to run benchmark"
        echo "                                           Testing will be truncated once maximum test duration has been reached"
        echo "                                           Disabled by default"
        echo "  --min-test-duration [MIN_TEST_DURATION]: Minimum duration in seconds to run benchmark"
        echo "                                           Images will be repeated until minimum test duration has been reached"
        echo "                                           Disabled by default"
        echo "  --multi-tile                           : Run benchmark in multi-tile configuration (default: '${MULTI_TILE}')"
        echo "  --num-inputs        [NUM_INPUTS]       : Number of images to load (default: '${NUM_INPUTS}')"
        echo "  --output-dir        [OUTPUT_DIR]       : Location to write output to. Required"
        echo "  --platform          [PLATFORM]         : Platform that inference is being ran on (default: '${PLATFORM}')"
        echo "                                           * CPU"
        echo "                                           * Flex"
        echo "                                           * CUDA"
        echo "                                           * Max"
	echo "  --ipex              [IPEX]             : Use Intel Extension for PyTorch for xpu device (default: '${IPEX}')"
        echo "  --precision         [PRECISION]        : Precision to use for the model (default: '${PRECISION}')"
        echo "                                           * bf16"
        echo "                                           * fp16"
        echo "                                           * fp32"
        echo "  --proxy             [PROXY]            : System proxy. Required to download models"
        echo "  --save              [SAVE_PATH]        : If specified model will be saved to this saved location (default: disabled)"
        echo "  --print-frequency   [PRINT_FREQUENCY]  : Number of seconds between status printouts (default: '${PRINT_FREQUENCY}')"
        echo "  --socket            [SOCKET]           : Socket to control telemetry capture (default: '${SOCKET}')"
        echo "  --streams           [STREAMS]          : Number of parallel streams to do inference on (default: '${STREAMS}')"
        echo "                                           Will be truncated to a multiple of BATCH_SIZE"
        echo "                                           If less than BATCH_SIZE will be increased to BATCH_SIZE"
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
    if [[ "${PLATFORM}" == "Max" ]]; then
        echo "Streams will be round-robin scheduled across multiple tiles"
        if [ $((STREAMS%2)) -ne 0 ]; then
        echo "WARNING: can't schedule evenly odd number of streams ($STREAMS) across tiles"
    fi
    fi
    if [[ "${PLATFORM}" == "Flex" ]]; then
        echo "ERROR: Flex does not support multitile"
        exit 1
    fi
    if [[ "${PLATFORM}" == "CUDA" ]]; then
        echo "ERROR: multitile is not implemented for CUDA"
        exit 1
    fi
fi

# Show test configuration
echo 'Running with parameters:'
echo " AMP:               ${AMP}"
echo " BATCH_SIZE:        ${BATCH_SIZE}"
echo " DATASET_DIR:       ${DATASET_DIR}"
echo " DUMMY:             ${DUMMY}"
echo " JIT:               ${JIT}"
echo " LOAD_PATH:         ${LOAD_PATH}"
echo " MAX_TEST_DURATION: ${MAX_TEST_DURATION}"
echo " MIN_TEST_DURATION: ${MIN_TEST_DURATION}"
echo " MODEL_NAME:        ${MODEL_NAME}"
echo " MULTI_TILE:        ${MULTI_TILE}"
echo " NUM_INPUTS:        ${NUM_INPUTS}"
echo " OUTPUT_DIR:        ${OUTPUT_DIR}"
echo " PLATFORM:          ${PLATFORM}"
echo " IPEX:              ${IPEX}"
echo " PRECISION:         ${PRECISION}"
echo " PRINT_FREQUENCY:   ${PRINT_FREQUENCY}"
echo " PROXY:             ${PROXY}"
echo " SAVE_PATH:         ${SAVE_PATH}"
echo " SOCKET:            ${SOCKET}"
echo " STREAMS:           ${STREAMS}"

# Set system proxies if requested.
if [[ "${PROXY}" != "" ]]; then
    export http_proxy=${PROXY}
    export https_proxy=${PROXY}
fi

# Set socket if sepecified
_socket_args=""
if [[ "${SOCKET}" != "" ]]; then
    _socket_args="--socket ${SOCKET}"
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
else
    echo "ERROR: Unknown precision '${PRECISION}' for model '${MODEL_NAME}'"
    exit 1
fi

# Specify if AMP should be used
if [[ ${AMP} == "no" ]]; then
    _amp_arg="--no-amp"
elif [[ ${AMP} == "yes" ]]; then
    _amp_arg=""
else
    echo "ERROR: Invalid value entered for 'AMP': ${AMP}"
    exit 1
fi

# Specify test duration if requested
_test_duration_args=""
if [[ ${MIN_TEST_DURATION} != "" ]]; then
    _test_duration_args="${_test_duration_args} --min-test-duration ${MIN_TEST_DURATION}"
fi
if [[ ${MAX_TEST_DURATION} != "" ]]; then
    _test_duration_args="${_test_duration_args} --max-test-duration ${MAX_TEST_DURATION}"
fi

# Specify if JIT should be used
if [[ ${JIT} == "none" ]]; then
    _jit_arg=""
elif [[ ${JIT} == "trace" ]]; then
    _jit_arg="--jit-trace"
elif [[ ${JIT} == "script" ]]; then
    _jit_arg="--jit-script"
elif [[ ${JIT} == "compile" ]]; then
    _jit_arg="--compile"
else
    echo "ERROR: Invalid value entered for 'JIT': ${JIT}"
    exit 1
fi

# General perf args
_perf_args="--pretrained --channels-last --no-grad"

# Create the output directory, if it doesn't already exist
mkdir -p $OUTPUT_DIR

# Set environment variables
_platform_args="--device ${PLATFORM}"
if [[ ${PLATFORM} == "Flex" ]]; then
    _platform_args="--device xpu"
    export IGC_EnableDPEmulation=1
    export CFESingleSliceDispatchCCSMode=1
    export IPEX_ONEDNN_LAYOUT=1
    export IPEX_LAYOUT_OPT=1
elif [[ ${PLATFORM} == "Max" ]]; then
    _platform_args="--device xpu"
    # Currently its an assumption that Max GPU uses these.
    export IGC_EnableDPEmulation=1
    export CFESingleSliceDispatchCCSMode=1
    export IPEX_ONEDNN_LAYOUT=1
    export IPEX_LAYOUT_OPT=1
elif [[ ${PLATFORM} == "CUDA" ]]; then
    _platform_args="--device cuda"
elif [[ ${PLATFORM} == "CPU" ]]; then
    _platform_args="--device cpu"
fi
export PROFILE="OFF"

if [[ "$IPEX" == "yes" ]]; then
    _platform_args+=" --ipex"
elif [[ "$IPEX" != "no" ]]; then
    echo "ERROR: Invalid value entered for 'IPEX': ${IPEX}"
    exit 1
fi

# Start inference script with numactl
echo "Starting inference..."
#TODO: Set ZE_AFFINITY_MASK for multiple tiles.
numactl --cpunodebind=0 --membind=0 python3 predict.py \
    ${_platform_args} \
    --arch ${MODEL_NAME} \
    ${_dataset_args} \
    --batch-size ${BATCH_SIZE} \
    --print-frequency ${PRINT_FREQUENCY} \
    --num-inputs ${NUM_INPUTS} \
    --width ${_img_width} --height ${_img_height} \
    ${_dtype_args} \
    ${_amp_arg} \
    ${_test_duration_args} \
    ${_jit_arg} \
    ${_perf_args} \
    ${_save_load_args} \
    ${_socket_args} \
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
