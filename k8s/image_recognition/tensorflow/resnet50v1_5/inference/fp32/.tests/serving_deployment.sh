@test "kubectl apply of serving resnet50v1-5-fp32-inference-k8s" {
  run $DETIK_CLIENT_NAME apply -f deployments/resnet50v1-5-fp32-inference-k8s/serving/serving.yaml
  (( $status == 0 ))
}

@test "deployment of resnet50v1-5-fp32-inference-k8s serving creates 1 pod and it is running" {
  run try "at most 4 times every 30s to get pods named 'resnet50v1-5-fp32-inference-k8s' and verify that 'status' is 'running'"
  (( $status == 0 ))
}

@test "kubectl delete of resnet50v1-5-fp32-inference-k8s serving" {
  run $DETIK_CLIENT_NAME delete -f deployments/resnet50v1-5-fp32-inference-k8s/serving/serving.yaml
  (( $status == 0 ))
}