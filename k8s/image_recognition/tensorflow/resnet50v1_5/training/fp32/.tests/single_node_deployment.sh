@test "kubectl apply of single-node resnet50v1-5-fp32-training-k8s" {
  run $DETIK_CLIENT_NAME apply -f deployments/resnet50v1-5-fp32-training-k8s/single-node/user-mounted-nfs.yaml
  echo "output = ${output}"
  (( $status == 0 ))
}

@test "single-node deployment of resnet50v1-5-fp32-training-k8s creates 1 pod and it is running" {
  try "at most 4 times every 30s to get pods named 'resnet50v1-5-fp32-training' and verify that 'status' is 'running'"
  echo "output = ${output}"
  (( $status == 0 ))
}

@test "verify job completion, check end of output includes text 'run_final'" {
  run launcher_logs
  echo "output = ${output}"
  (( $status == 0 ))
  [[ ${lines[4]} =~ run_final ]]
}

@test "kubectl delete of single-node resnet50v1-5-fp32-training-k8s" {
  run $DETIK_CLIENT_NAME delete -f deployments/resnet50v1-5-fp32-training-k8s/single-node/user-mounted-nfs.yaml
  echo "output = ${output}"
  (( $status == 0 ))
}