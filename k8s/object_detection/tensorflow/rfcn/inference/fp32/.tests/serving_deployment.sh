@test "kubectl apply of rfcn-fp32-inference-k8s serving" {
  run $DETIK_CLIENT_NAME apply -f deployments/rfcn-fp32-inference-k8s/serving/user-mounted-nfs.yaml
  (( $status == 0 ))
}

@test "deployment of rfcn-fp32-inference-k8s serving creates 1 pod and it is running" {
  run try "at most 4 times every 30s to get pods named 'rfcn-fp32-inference-k8s' and verify that 'status' is 'running'"
  (( $status == 0 ))
}

@test "kubectl delete of rfcn-fp32-inference-k8s serving" {
  run $DETIK_CLIENT_NAME delete -f deployments/rfcn-fp32-inference-k8s/serving/user-mounted-nfs.yaml
  (( $status == 0 ))
}