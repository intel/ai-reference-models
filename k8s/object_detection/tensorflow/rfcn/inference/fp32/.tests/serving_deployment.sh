@test "kubectl apply of rfcn-fp32-inference-k8s serving" {
  run $DETIK_CLIENT_NAME apply -f deployments/rfcn-fp32-inference-k8s/serving/user-mounted-nfs.yaml
  echo "output = ${output}"
  (( $status == 0 ))
}

@test "deployment of rfcn-fp32-inference-k8s serving creates 1 pod and it is running" {
  try "at most 4 times every 30s to get pods named 'rfcn-fp32-inference' and verify that 'status' is 'running'"
  echo "output = ${output}"
  (( $status == 0 ))
}

@test "verify job completion, check end of output includes text 'run_final'" {
  run launcher_logs
  echo "output = ${output}"
  (( $status == 0 ))
  [[ ${lines[4]} =~ run_final ]]
}

@test "kubectl delete of rfcn-fp32-inference-k8s serving" {
  run $DETIK_CLIENT_NAME delete -f deployments/rfcn-fp32-inference-k8s/serving/user-mounted-nfs.yaml
  echo "output = ${output}"
  (( $status == 0 ))
}