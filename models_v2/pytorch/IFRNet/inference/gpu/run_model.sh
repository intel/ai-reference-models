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

set -e
SCRIPT_DIR=$(dirname ${BASH_SOURCE[0]})

# Specify default arguments
[[ "${DATASET_DIR}" == "" ]]    && DATASET_DIR=""
[[ "${MODELS_DIR}" == "" ]]     && MODELS_DIR="models"
[[ "${LOAD_PATH}" == "" ]]      && LOAD_PATH=""
[[ "${DUMMY}" == "" ]]          && DUMMY="no"
[[ "${DATA_CHANNELS}" == "" ]]  && DATA_CHANNELS=3
[[ "${DATA_HEIGHT}" == "" ]]    && DATA_HEIGHT=720
[[ "${DATA_WIDTH}" == "" ]]     && DATA_WIDTH=1280
[[ "${NUM_INPUTS}" == "" ]]     && NUM_INPUTS=""
[[ "${AMP}" == "" ]]            && AMP="no"
[[ "${PRECISION}" == "" ]]      && PRECISION=""
[[ "${STREAMS}" == "" ]]        && STREAMS=""
[[ "${ASYNC}" == "" ]]          && ASYNC=""
[[ "${SAVE_IMAGES}" == "" ]]    && SAVE_IMAGES="no"
[[ "${MIN_PSNR_DB}" == "" ]]    && MIN_PSNR_DB=25
[[ "${MIN_PASS_PCT}" == "" ]]   && MIN_PASS_PCT=95
[[ "${WARMUP}" == "" ]]         && WARMUP=10
[[ "${INTERPOLATION}" == "" ]]  && INTERPOLATION=""
[[ "${PLATFORM}" == "" ]]       && PLATFORM="Flex"
[[ "${OUTPUT_DIR}" == "" ]]     && OUTPUT_DIR=$(pwd)/output-dir
[[ "${MIN_TEST_DURATION}" == "" ]] && MIN_TEST_DURATION=""
[[ "${MAX_TEST_DURATION}" == "" ]] && MAX_TEST_DURATION=""
[[ "${SOCKET}" == "" ]]         && SOCKET=""
[[ "${IPEX}" == "" ]]           && IPEX="yes"

# Process CLI arguments as overides for environment variables
VALID_ARGS=$(getopt -o h --long data:,modelsdir:,pretrained-weights:,dummy,data-channels:,data-height:,data-width:,help,num-inputs:,amp,precision:,async:,streams:,psnr-threshold:,saveimages,min-pass-pct:,min-test-duration:,max-test-duration:,socket:,platform:,warmup:,interpolation:,ipex: -- "$@")
if [[ $? -ne 0 ]]; then
    exit 1;
fi
eval set -- "$VALID_ARGS"
while [ : ]; do
  case "$1" in
    --data)
        DATASET_DIR="$2"
        shift 2
        ;;
    --modelsdir)
        MODELS_DIR="$2"
        shift 2
        ;;
    --dummy)
        DUMMY="yes"
        shift 1
        ;;
    --data-channels)
        DATA_CHANNELS="$2"
        shift 2
        ;;
    --data-height)
        DATA_HEIGHT="$2"
        shift 2
        ;;
    --data-width)
        DATA_WIDTH="$2"
        shift 2
        ;;
    --pretrained-weights)
        LOAD_PATH="$2"
        shift 2
        ;;
    --saveimages)
        SAVE_IMAGES="yes"
        shift 1
        ;;
    --num-inputs)
        NUM_INPUTS=$2
        shift 2
        ;;
    --async)
        ASYNC=$2
        shift 2
        ;;
    --streams)
        STREAMS=$2
        shift 2
        ;;
    --precision)
        PRECISION=$2
        shift 2
        ;;
    --amp)
        AMP="yes"
        shift 1
        ;;
    --output-dir)
        OUTPUT_DIR=$2
        shift 2
        ;;
    --device)
        PLATFORM=$2
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
    --psnr-threshold)
        MIN_PSNR_DB=$2
        shift 2
        ;;
    --min-pass-pct)
        MIN_PASS_PCT=$2
        shift 2
        ;;
    --min-test-duration)
        MIN_TEST_DURATION=$2
        shift 2
        ;;
    --max-test-duration)
        MAX_TEST_DURATION=$2
        shift 2
        ;;
    --socket)
        SOCKET=$2
        shift 2
        ;;
    --warmup)
        WARMUP=$2
        shift 2
        ;;
    --interpolation)
        INTERPOLATION=$2
        shift 2
        ;;
    --proxy)
        PROXY=$2
        shift 2
        ;;
    -h | --help)
        echo "Usage: $(basename $0)"
        echo "  --data                    [DATASET_DIR] : Location to load images from (default: '${DATASET_DIR}')"
        echo "  --modelsdir                [MODELS_DIR] : Location to load models from (default: 'model')"
        echo "  --dummy                                 : Use randomly generated dummy dataset in place of dataset (default: disabled)"
        echo "  --data-channels         [DATA_CHANNELS] : Number of color channels of randomly generated dataset (default: '${DATA_CHANNELS}')"
        echo "  --data-height             [DATA_HEIGHT] : Height of images in randomly generated dataset (default: '${DATA_HEIGHT}')"
        echo "  --data-width               [DATA_WIDTH] : Width of images in randomly generated dataset (default: '${DATA_WIDTH}')"
        echo "  --pretrained-weights        [LOAD_PATH] : If specified model will be loaded from specified location (default: disabled)"
        echo "  --num-inputs                   [FRAMES] : Number of pairs of input frames per iteration (default: '${NUM_INPUTS}')"
        echo "  --output-dir               [OUTPUT_DIR] : Location to write output to (default $(pwd)/output-dir)"
        echo "  --async                         [ASYNC] : Number of batches after which to issue a gpu sync. Default=0: meaning after all workloads requested"
        echo "  --streams                     [STREAMS] : Number of parallel processes/streams to run interpolation in. Default=1"
        echo "  --precision                 [PRECISION] : Datatype to use for model and input tensors to run inference. Choices: fp16 (default) | bf16 | fp32"
        echo "  --amp                                   : Enable Autocast feature in Pytorch to allow mixed precision model and input tensors. Default disabled"
        echo "  --platform                   [PLATFORM] : Platform that inference is being ran on (default: '${PLATFORM}')"
        echo "                                            * CPU"
        echo "                                            * Flex"
        echo "                                            * CUDA"
        echo "                                            * Max"
        echo "  --ipex                           [IPEX] : Use Intel Extension for PyTorch for xpu device (default: '${IPEX}')"
        echo "  --warmup                       [WARMUP] : Number of frames to be run as warmup"
        echo "  --interpolation         [INTERPOLATION] : Number of interpolated frames to generate per input pair (default: '${INTERPOLATION}')"
        echo "  --saveimages                            : Save interpolated frames in output directory  (default: disabled)"
        echo "  --psnr-threshold          [MIN_PSNR_DB] : Min PSNR in dB to consider one inference submission a pass (default 25dB)"
        echo "  --min-pass-pct           [MIN_PASS_PCT] : Min % of frames to pass to consider the overall run a pass (default 95)"
        echo "  --socket                       [SOCKET] : Socket to control telemetry capture: default ''"
        echo "  --min-test-duration [MIN_TEST_DURATION] : Minimum time in seconds in to run the test. Only applies to --dummy, and overrides --num-inputs"
        echo "  --max-test-duration [MIN_TEST_DURATION] : Maximum time in seconds in to run the test. Only applies to --dummy, and overrides --num-inputs"
        echo ""
        echo "NOTE: Arguments may also be specified through command line variables using the name in '[]'."
        echo "      For example 'export PLATFORM=Flex'."
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
        args="--dummy"
    else
        # Ambiguous case. Terminate.
        echo "ERROR: Cannot handle simultaneous usage of '--dummy' and '--data'/'DATASET_DIR'."
        exit 1
    fi
else
    if [[ -z "${DATASET_DIR}" ]]; then
        # Ambiguous case. Terminate.
        echo "ERROR: Dataset must be provided or --dummy must be enabled:"
        echo "ERROR: add '--dummy' or '--data [DATASET_DIR]'."
        exit 1
    else
        if [[ -d "${DATASET_DIR}" ]]; then
            # Actual dataset provided provided. OK case.
            args="--data ${DATASET_DIR}"
        else
            echo "ERROR: The requested dataset '${DATASET_DIR}' does not exist!"
            exit 1
        fi
    fi

    if [[ ${SAVE_IMAGES} == "yes" ]]; then
        args="$args --saveimages"
    fi
fi

# Show test configuration
echo 'Running with parameters:'
echo " AMP:               ${AMP}"
echo " BATCH_SIZE:        1"
echo " DATASET_DIR:       ${DATASET_DIR}"
echo " DUMMY:             ${DUMMY}"
echo " DATA_CHANNELS:     ${DATA_CHANNELS}"
echo " DATA_HEIGHT:       ${DATA_HEIGHT}"
echo " DATA_WIDTH:        ${DATA_WIDTH}"
echo " INTERPOLATION:     ${INTERPOLATION}"
echo " LOAD_PATH:         ${LOAD_PATH}"
echo " MIN_TEST_DURATION: ${MIN_TEST_DURATION}"
echo " MAX_TEST_DURATION: ${MAX_TEST_DURATION}"
echo " NUM_INPUTS:        ${NUM_INPUTS}"
echo " OUTPUT_DIR:        ${OUTPUT_DIR}"
echo " SAVE_IMAGES:       ${SAVE_IMAGES}"
echo " SOCKET:            ${SOCKET}"
echo " STREAMS:           ${STREAMS}"
echo " PLATFORM:          ${PLATFORM}"
echo " IPEX:              ${IPEX}"
echo " PRECISION:         ${PRECISION}"
echo " PROXY:             ${PROXY}"

# Set system proxies if requested.
if [[ "${PROXY}" != "" ]]; then
    export http_proxy=${PROXY}
    export https_proxy=${PROXY}
fi

# First download+install model and weights if directories indicating their availability don't exist
${SCRIPT_DIR}/./get_model.sh ${MODELS_DIR} ${LOAD_PATH}
export PYTHONPATH=${PYTHONPATH}:${MODELS_DIR}

# Configure save and load params
if [[ ${LOAD_PATH} != "" ]]; then
    args="${args} --pretrained-weights ${LOAD_PATH}"
fi

# Get platform/device
[[ "${PLATFORM}" == "CPU" ]] && PLATFORM_ARGS=cpu
[[ "${PLATFORM}" == "Flex" ]] && PLATFORM_ARGS=xpu
[[ "${PLATFORM}" == "Max" ]] && PLATFORM_ARGS=xpu
[[ "${PLATFORM}" == "CUDA" ]] && PLATFORM_ARGS=cuda
args="${args} --device ${PLATFORM_ARGS}"

# Create the output directory, if it doesn't already exist
mkdir -p ${OUTPUT_DIR}

# Set environment variables
if [[ ${PLATFORM} == "Flex" ]]; then
    export IGC_EnableDPEmulation=1
    export CFESingleSliceDispatchCCSMode=1
    export IPEX_ONEDNN_LAYOUT=1
    export IPEX_LAYOUT_OPT=1
elif [[ ${PLATFORM} == "Max" ]]; then
    # Currently its an assumption that Max uses these.
    export IGC_EnableDPEmulation=1
    export CFESingleSliceDispatchCCSMode=1
    export IPEX_ONEDNN_LAYOUT=1
    export IPEX_LAYOUT_OPT=1
fi

if [[ "$IPEX" == "yes" ]]; then
    args="${args} --ipex"
elif [[ "$IPEX" == "no" ]]; then
    # IFRNet hardcodes usage of xpu and cuda if available in few places,
    # so even PLATFORM=CPU will have some operations executed on GPU if
    # the one is found. So, we need to fully enable fp64 emulation support
    # for XPU, hence setting BOTH environment variables below (one for
    # runtime, another for compiler).

    # This setting is required on native XPU backend, but might
    # give lower performance on IPEX
    export OverrideDefaultFP64Settings=1
    export IGC_EnableDPEmulation=1
else
    echo "ERROR: Invalid value entered for 'IPEX': ${IPEX}"
    exit 1
fi

# Test duration related inputs
[[ ${NUM_INPUTS} != "" ]] && args="${args} --num-inputs ${NUM_INPUTS}"
[[ ${MIN_TEST_DURATION} != "" ]] && args="${args} --min-test-duration ${MIN_TEST_DURATION}"
[[ ${MAX_TEST_DURATION} != "" ]] && args="${args} --max-test-duration ${MAX_TEST_DURATION}"

# Data dims related input
args="${args} --datadim ${DATA_CHANNELS} ${DATA_HEIGHT} ${DATA_WIDTH}"

# PSNR threshold for pass fail judgement
args="${args} --psnr-threshold ${MIN_PSNR_DB} --min-pass-pct ${MIN_PASS_PCT}"

# Additional Args
[[ ${INTERPOLATION} != "" ]] && args="${args} --interpolation ${INTERPOLATION}"
[[ ${ASYNC} != "" ]] && args="${args} --async ${ASYNC}"
[[ ${STREAMS} != "" ]] && args="${args} --streams ${STREAMS}"
[[ ${PRECISION} != "" ]] && args="${args} --precision ${PRECISION}"
[[ ${AMP} == "yes" ]] && args="${args} --amp"

# Socket for Telemetry capture
[[ ${SOCKET} != "" ]] && args="${args} --socket ${SOCKET}"

CMDLINE="python3 main.py ${args} --output-dir ${OUTPUT_DIR}"
echo ${CMDLINE} > ${OUTPUT_DIR}/output_raw.log

# Start inference script with numactl
echo "Starting inference..."
numactl --cpunodebind=0 ${CMDLINE}  2>&1 | tee -a ${OUTPUT_DIR}/output_raw.log
exit_code=${PIPESTATUS[0]}
echo "Inference complete"

echo "Output directory contents:"
ls ${OUTPUT_DIR}

if [ $exit_code -ne 0 ]; then
    echo "ERROR: Model scripts terminated with non-zero exit code: $exit_code"
    exit 1
fi
exit 0
