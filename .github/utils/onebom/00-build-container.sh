#!/bin/bash

source "./shared_vars.sh"

# build the requirements container with just pre-reqs
docker build \
  --build-arg https_proxy \
  --build-arg http_proxy \
  --build-arg HTTPS_PROXY \
  --build-arg HTTP_PROXY \
  --build-arg no_proxy \
  --build-arg NO_PROXY \
  -t ${PREREQS_CONTAINER_NAME} . 

check_error $? "Build PreReq Container  FAIL"
