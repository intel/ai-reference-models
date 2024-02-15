#!/bin/bash

#This will replace the requirements with locked versions

source "./shared_vars.sh"
debug "HOST_PATH=${HOST_PATH}"

docker exec \
  "${RUNNING_CONTAINER_NAME}" \
  "${HOST_PATH}/${SCRIPT_ROOT}/replace-requirements-with-locked-versions.sh"
check_error $? "Error replacing locked verions"
