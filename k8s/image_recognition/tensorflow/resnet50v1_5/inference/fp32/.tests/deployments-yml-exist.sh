#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
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

load "lib/test_helper"

@test "validate package for resnet50v1-5-fp32-inference-k8s in framework k8s creates model package file" {
  run model-builder package -f k8s resnet50v1-5-fp32-inference-k8s
  (( $status == 0 ))
  [[ -f output/resnet50v1-5-fp32-inference-k8s.tar.gz ]]
}

@test "validate generate-deployment for resnet50v1-5-fp32-inference-k8s in framework k8s creates deployment files under deployments/resnet50v1-5-fp32-training-k8s/" {
  run model-builder generate-deployment -f k8s resnet50v1-5-fp32-inference-k8s
  (( $status == 0 ))
  [[ -f deployments/resnet50v1-5-fp32-inference-k8s/serving/serving.yaml ]]
  (( $(last_modified deployments/resnet50v1-5-fp32-inference-k8s/serving/serving.yaml) <= 50 ))
}