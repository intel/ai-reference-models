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