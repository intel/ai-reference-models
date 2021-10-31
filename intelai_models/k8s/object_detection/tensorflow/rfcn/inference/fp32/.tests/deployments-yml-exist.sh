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

@test "validate package for rfcn-fp32-inference-k8s in framework k8s creates model package file" {
  run model-builder package -f k8s rfcn-fp32-inference-k8s
  (( $status == 0 ))
  [[ -f output/rfcn-fp32-inference-k8s.tar.gz ]]
}

@test "validate generate-deployment for rfcn-fp32-inference-k8s in framework k8s creates deployment files under deployments/rfcn-fp32-inference-k8s/" {
  run model-builder generate-deployment -f k8s rfcn-fp32-inference-k8s
  (( $status == 0 ))
  [[ -f deployments/rfcn-fp32-inference-k8s/pipeline/user-allocated-pvc.yaml ]]
  [[ -f deployments/rfcn-fp32-inference-k8s/pipeline/user-mounted-nfs.yaml ]]
  [[ -f deployments/rfcn-fp32-inference-k8s/serving/user-allocated-pvc.yaml ]]
  [[ -f deployments/rfcn-fp32-inference-k8s/serving/user-mounted-nfs.yaml ]]
  (( $(last_modified deployments/rfcn-fp32-inference-k8s/pipeline/user-allocated-pvc.yaml) <= 50 ))
  (( $(last_modified deployments/rfcn-fp32-inference-k8s/pipeline/user-mounted-nfs.yaml) <= 50 ))
  (( $(last_modified deployments/rfcn-fp32-inference-k8s/serving/user-allocated-pvc.yaml) <= 50 ))
  (( $(last_modified deployments/rfcn-fp32-inference-k8s/serving/user-mounted-nfs.yaml) <= 50 ))
}