#!/bin/bash
# This Script will generate the trivy scans after all requirements have been locked

source "./shared_vars.sh"
debug "HOST_PATH=${HOST_PATH}"

docker exec \
  "${RUNNING_CONTAINER_NAME}" \
  "${HOST_PATH}/${SCRIPT_ROOT}/generate-trivy-scans.sh"
check_error $? "Error generating trivy scans"
