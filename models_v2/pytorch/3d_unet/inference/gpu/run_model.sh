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
[[ "${DUMMY}" == "" ]]          && DUMMY="yes"
[[ "${JIT}" == "" ]]            && JIT="trace"
[[ "${MULTI_TILE}" == "" ]]     && MULTI_TILE="False"
[[ "${NUM_IMAGES}" == "" ]]     && NUM_IMAGES=1
[[ "${NUM_ITERATIONS}" == "" ]] && NUM_ITERATIONS=10
[[ "${PRECISION}" == "" ]]      && PRECISION="fp16"
[[ "${STATUS_PRINTS}" == "" ]]  && STATUS_PRINTS=10
[[ "${STREAMS}" == "" ]]        && STREAMS=1
[[ "${IPEX}" == "" ]]           && IPEX="yes"

# Process CLI arguments as overides for environment variables
VALID_ARGS=$(getopt -o h --long amp:,batch-size:,data:,dummy,help,jit:,multi-tile,num-images:,num-iterations:,output-dir:,platform:,precision:,status-prints:,streams:,ipex: -- "$@")
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
    --batch-size)
        BATCH_SIZE=$2
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
    --ipex)
        IPEX=$2
        shift 2
        ;;
    --precision)
        PRECISION=$2
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
        echo "  --batch-size     [BATCH_SIZE]    : Batch size to use (default: '${BATCH_SIZE}')"
        echo "  --dummy                          : Use randomly generated dummy dataset in place of '--data' argument (default: disabled)"
        echo "  --jit            [JIT]           : JIT method to use (default: '${JIT}')"
        echo "                                     * none"
        echo "                                     * compile"
        echo "                                     * script"
        echo "                                     * trace"
        echo "  --multi-tile                     : Run benchmark in multi-tile configuration (default: '${MULTI_TILE}')"
        echo "  --num-images     [NUM_IMAGES]    : Number of images to load (default: '${NUM_IMAGES}')"
        echo "  --num-iterations [NUM_ITERATIONS]: Number of times to test each batch (default: '${NUM_ITERATIONS}')"
        echo "  --output-dir     [OUTPUT_DIR]    : Location to write output to. Required"
        echo "  --platform       [PLATFORM]      : Platform that inference is being ran on (default: '${PLATFORM}')"
        echo "                                     * CPU"
        echo "                                     * CUDA"
        echo "                                     * Flex"
        echo "                                     * Max"
        echo "  --ipex           [IPEX]          : Use Intel Extension for PyTorch for xpu device (default: '${IPEX}')"
        echo "  --precision      [PRECISION]     : Precision to use for the model (default: '${PRECISION}')"
        echo "                                     * bf16"
        echo "                                     * fp16"
        echo "                                     * fp32"
        echo "                                     * int8"
        echo "  --status-prints  [STATUS_PRINTS] : Total number of status messages to display during inference benchmarking (default: '${STATUS_PRINTS}')"
        echo "  --streams        [STREAMS]       : Number of parallel streams to do inference on (default: '${STREAMS}')"
        echo "                                     Will be truncated to a multiple of BATCH_SIZE"
        echo "                                     If less than BATCH_SIZE will be increased to BATCH_SIZE"
        echo ""
        echo "NOTE: Arguments may also be specified through command line variables using the name in '[]'."
        echo "      For example 'export NUM_IMAGES=16'."
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
    _dataset_args="--dummy"
else
    # Preprocessed dataset location is handled during setup script and python scripts have pre defined locations
    _dataset_args=""
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
echo " AMP:              ${AMP}"
echo " BATCH_SIZE:       ${BATCH_SIZE}"
echo " DUMMY:            ${DUMMY}"
echo " JIT:              ${JIT}"
echo " MULTI_TILE:       ${MULTI_TILE}"
echo " NUM_ITERATIONS:   ${NUM_ITERATIONS}"
echo " NUM_IMAGES:       ${NUM_IMAGES}"
echo " OUTPUT_DIR:       ${OUTPUT_DIR}"
echo " STATUS_PRINTS:    ${STATUS_PRINTS}"
echo " STREAMS:          ${STREAMS}"
echo " PLATFORM:         ${PLATFORM}"
echo " PRECISION:        ${PRECISION}"

# known issue for multitile
if [[ "${MULTI_TILE}" == "True" ]]; then
    export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
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
    echo "ERROR: Precision '${PRECISION}' is not supported yet for model"
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

# Specify if JIT should be used
if [[ ${JIT} == "none" ]]; then
    _jit_arg="--use-jit none"
elif [[ ${JIT} == "trace" ]]; then
    _jit_arg="--use-jit trace"
elif [[ ${JIT} == "script" ]]; then
    _jit_arg="--use-jit script"
elif [[ ${JIT} == "compile" ]]; then
    _jit_arg="--use-jit compile"
else
    echo "ERROR: Invalid value entered for 'JIT': ${JIT}"
    exit 1
fi

# General perf args
_perf_args="--channels-last"

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
    # Currently its an assumption that PCV uses these.
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

# Check if preprocessing has been done
if [[ ${DUMMY} == "no" ]] && [[ ! -d ${DATASET_DIR}/build/preprocessed_data ]]; then
    ./preprocess.sh
fi

# Start inference script with numactl
echo "Starting inference..."
#TODO: Set ZE_AFFINITY_MASK for multiple tiles.
numactl --cpunodebind=0 --membind=0 python3 predict.py \
    ${_platform_args} \
    ${_dataset_args} \
    --batch-size ${BATCH_SIZE} \
    --status-prints ${STATUS_PRINTS} \
    --max-val-dataset-size ${NUM_IMAGES} \
    --batch-streaming ${NUM_ITERATIONS} \
    ${_dtype_args} ${_amp_arg} ${_jit_arg} ${_perf_args} \
    --warm-up 10 \
    --output-dir ${OUTPUT_DIR} \
    --total-instances ${STREAMS} \
    --terminate-if-sync-fail \
    --data ${DATASET_DIR}/build/preprocessed_data\
    --label-data-dir ${DATASET_DIR}/build/raw_data/nnUNet_raw_data/Task043_BraTS2019/labelsTr\
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
