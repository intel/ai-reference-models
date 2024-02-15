#!/bin/bash

# this script just launches the container with env vars and mounts

source "./shared_vars.sh"
debug "HOST_PATH=${HOST_PATH}"
HOST_MOUNT_POINT="$(cd ${REF_MODELS_ROOT} && pwd)"
debug "HOST_MOUNT_POINT: ${HOST_MOUNT_POINT}"

docker run \
  -d \
  --disable-content-trust \
  --name="${RUNNING_CONTAINER_NAME}" \
  -e https_proxy \
  -e http_proxy \
  -e HTTPS_PROXY \
  -e HTTP_PROXY \
  -e no_proxy \
  -e NO_PROXY \
  -e HOST_PATH="${HOST_PATH}" \
  -e SCRIPT_ROOT="${SCRIPT_ROOT}" \
  -v ${HOST_MOUNT_POINT}:${HOST_PATH} \
  -w ${HOST_PATH} \
  ${PREREQS_CONTAINER_NAME}

#sleep and wait for it to start
retVal=1
while [[ $retVal -ne 0 ]]
do
  echo "Waiting for SBOM Container..."
  sleep 1
  docker ps | grep "${RUNNING_CONTAINER_NAME}"
  retVal=$?
done
