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
FROM python:3.9

WORKDIR /data_connector
COPY ./src .

COPY ./data_connector_requirements.txt .
COPY  ./data_connector_azure_test_requirements.txt .
RUN pip install --no-cache-dir -r data_connector_azure_test_requirements.txt

RUN python -m pytest 
