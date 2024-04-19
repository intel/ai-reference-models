#!/bin/env bash
#
# Copyright (c) 2023 Intel Corporation
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
#

export DOCKER_BUILD_ARGS="--build-arg ftp_proxy=${ftp_proxy} --build-arg FTP_PROXY=${FTP_PROXY} --build-arg http_proxy=${http_proxy} --build-arg HTTP_PROXY=${HTTP_PROXY} --build-arg https_proxy=${https_proxy} --build-arg HTTPS_PROXY=${HTTPS_PROXY} --build-arg no_proxy=${no_proxy} --build-arg NO_PROXY=${NO_PROXY} --build-arg socks_proxy=${socks_proxy} --build-arg SOCKS_PROXY=${SOCKS_PROXY}"
#export DOCKER_BUILD_ARGS="--build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy}"

export DOCKER_RUN_ENVS="--env ftp_proxy=${ftp_proxy} --env FTP_PROXY=${FTP_PROXY} --env http_proxy=${http_proxy} --env HTTP_PROXY=${HTTP_PROXY} --env https_proxy=${https_proxy} --env HTTPS_PROXY=${HTTPS_PROXY} --env no_proxy=${no_proxy} --env NO_PROXY=${NO_PROXY} --env socks_proxy=${socks_proxy} --env SOCKS_PROXY=${SOCKS_PROXY}"

VERSION=4.0
export IMAGE_NAME=mlperf_inference_gptj_int4:${VERSION}



echo "Building GPT-J INT4 workflow container"
DOCKER_BUILDKIT=1 docker build  --no-cache --progress=plain ${DOCKER_BUILD_ARGS} -f ../../../../../docker/pyt-cpu/gpt-j_mlperf-inference/Dockerfile_int4 -t ${IMAGE_NAME} ../../../../


#docker run --rm -it ${IMAGE_NAME} python -c "import torch; print('torch:', torch.__version__)"

# docker push intel/intel-optimized-pytorch:v1.7.0-ipex-v1.2.0-bert
