#!/bin/bash

(( EUID != 0 )) && echo "fatal: must be executed under root" && exit 1

OUTPUT_DIR=${OUTPUT_DIR:-$PWD}

PROFILE=${PROFILE:-profiles/b0.bf16.csv}

PLATFORM=${PLATFORM:-Flex}
AMP=${AMP:-yes}
JIT=${JIT:-trace}

if [[ ${PLATFORM} == "Flex" || ${PLATFORM} == "Max" ]]; then
  IMAGE=${IMAGE:-intel/image-recognition:pytorch-flex-gpu-efficientnet-inference}
  platform_opt="--device /dev/dri/"
elif [[ ${PLATFORM} == "CUDA" ]]; then
  IMAGE=${IMAGE:-intel/image-recognition:pytorch-cuda-gpu-efficientnet-inference}
  platform_opt="--gpus all"
else
  echo "fatal: unsupported platform: ${PLATFORM}"
  exit 1
fi

METADATA="$METADATA"
METADATA+=" config.system.docker0.docker.image=$IMAGE"
METADATA+=" config.system.docker0.docker.sha256=$(docker images --no-trunc --quiet $IMAGE | cut -d: -f2)"

# arguments in {} are mapped by benchmark.py from:
# 1. Profile given by --profile argument for each iteration of the test
# 2. Predefined variables: {output_dir}, {socket}
python3 -m benchmark --profile=$PROFILE --output_dir=$OUTPUT_DIR --telemetry --socket=/tmp/telemetry.s --platform=$PLATFORM --indent 4 --metadata "$METADATA" \
  docker run -it --rm --ipc=host --cap-add SYS_NICE $platform_opt \
    $(env | grep -E '(_proxy=|_PROXY)' | sed 's/^/-e /') \
    -e PLATFORM=${PLATFORM} \
    -e AMP=${AMP} \
    -e JIT=${JIT} \
    -e MODEL_NAME={model_name} \
    -e BATCH_SIZE={batch_size} \
    -e NUM_IMAGES={num_images} \
    -e STREAMS={streams} \
    -e NUM_ITERATIONS={num_iterations} \
    -e PRECISION={precision} \
    -e OUTPUT_DIR=/opt/output \
    -v {output_dir}:/opt/output \
    -e SOCKET=/tmp/telemetry.s \
    -v {socket}:/tmp/telemetry.s \
    $IMAGE \
      /bin/bash -c './run_model.sh --dummy'
