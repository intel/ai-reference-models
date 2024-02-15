#!/bin/bash

# this script just kills the container 

source "./shared_vars.sh"

docker stop "${RUNNING_CONTAINER_NAME}"
docker rm "${RUNNING_CONTAINER_NAME}"
