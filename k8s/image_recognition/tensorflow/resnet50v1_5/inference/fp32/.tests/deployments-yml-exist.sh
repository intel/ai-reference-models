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