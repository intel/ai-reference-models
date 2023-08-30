#
# -*- coding: utf-8 -*-
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

#

PYT_BASE_IMAGE=intel/intel-extension-for-pytorch
PYT_BASE_TAG=2.0.0-pip-base
all:
	@PYT_BASE_IMAGE=${PYT_BASE_IMAGE} \
	 PYT_BASE_TAG=${PYT_BASE_TAG} \
	 docker compose -f pyt-cpu-docker-compose.yml up --build
