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

@test "kubectl apply of single-node bert-large-fp32-training-k8s" {
  run $DETIK_CLIENT_NAME apply -f deployments/bert-large-fp32-training-k8s/single-node/user-mounted-nfs.yaml
  echo "output = ${output}"
  (( $status == 0 ))
}

@test "single-node deployment of bert-large-fp32-training-k8s creates 1 pod and it is running" {
  try "at most 4 times every 30s to get pods named 'bert-large-fp32-training' and verify that 'status' is 'running'"
  echo "output = ${output}"
  (( $status == 0 ))
}

@test "verify job completion, check end of output includes text 'run_final'" {
  run launcher_logs
  echo "output = ${output}"
  (( $status == 0 ))
  [[ ${lines[4]} =~ run_final ]]
}

@test "kubectl delete of single-node bert-large-fp32-training-k8s" {
  run $DETIK_CLIENT_NAME delete -f deployments/bert-large-fp32-training-k8s/single-node/user-mounted-nfs.yaml
  echo "output = ${output}"
  (( $status == 0 ))
}