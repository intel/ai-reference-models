@test "kubectl apply of single-node resnet50v1-5-fp32-training-k8s" {
  run $DETIK_CLIENT_NAME apply -f deployments/resnet50v1-5-fp32-training-k8s/single-node/user-mounted-nfs.yaml
  (( $status == 0 ))
}

@test "single-node deployment of resnet50v1-5-fp32-training-k8s creates 1 pod and it is running" {
  run try "at most 4 times every 30s to get pods named 'resnet50v1-5-fp32-training-k8s' and verify that 'status' is 'running'"
  (( $status == 0 ))
}

@test "kubectl delete of single-node resnet50v1-5-fp32-training-k8s" {
  run $DETIK_CLIENT_NAME delete -f deployments/resnet50v1-5-fp32-training-k8s/single-node/user-mounted-nfs.yaml
  (( $status == 0 ))
}