#!/bin/bash

# Ouput:
# When this script runs successfully, there will be a file that lists all of the 
# locked pypi packages with versions. 

source "shared_vars.sh"
debug "HOST_PATH=${HOST_PATH}"

# install all requirements in isolated environment
docker exec "${RUNNING_CONTAINER_NAME}" ls -la /host
docker exec "${RUNNING_CONTAINER_NAME}" "${HOST_PATH}/${SCRIPT_ROOT}/resolve-and-freeze-reqs.sh"
check_error $? "Error locking collated requirements in docker container"
