# A Helm chart for TensorFlow Serving

Prior to running this how-to, please ensure you have `Helm 3` installed per instructions on this page:

https://helm.sh/docs/intro/install/

To install a Helm chart that deploys a TensorFlow Serving, try this:
```shell
# Note: Ensure a namespace with the following name exists:
export SERVING_NAMESPACE=<NAME_SPACE_TO_DEPLOY_CHART>   # tensorflow-serving
export CHART_NAME=<NAME_OF_DEPLOYD_CHART>   # resnet50v1-5
# Navigate to the folder where the desired chart is located:
cd tools/k8s/devops/helm-charts/tensorflow-serving

# Install a Serving chart
helm install \
     --namespace $SERVING_NAMESPACE \
     --debug $CHART_NAME . \
     --set service.internalPort=8500 \
     --set podSecurityContext.fsGroup=<YOUR_FS_GROUP> \
     --set podSecurityContext.runAsGroup=<YOUR_DESIRED_GROUP> \
     --set podSecurityContext.runAsUser=<YOUR_USER_ID> \
     --set model_base_path=<MODEL_BASE_PATH_ENV_VALUE> \
     --set model_name=<MODEL_NAME_ENV_VALUE> \
     --set models_path=<MODEL_LOCAL_PATH> \
     --set replicaCount=<SERVING_POD_REPLICAS>
```

This returns something like this:
```shell
NOTES:
1. Get the application URL by running these commands:
  export POD_NAME=$(kubectl get pods --namespace tensorflow-serving -l "app.kubernetes.io/name=tensorflow-serving,app.kubernetes.io/instance=resnet50v1-5" -o jsonpath="{.items[0].metadata.name}")
  export CONTAINER_PORT=$(kubectl get pod --namespace tensorflow-serving $POD_NAME -o jsonpath="{.spec.containers[0].ports[0].containerPort}")
  echo "Visit http://127.0.0.1:8501 to use your application"
  kubectl --namespace tensorflow-serving port-forward $POD_NAME 8501:$CONTAINER_PORT
```

Once run the above commands prompted by NOTES, on another shell, try the following:

```shell
# Clone TensorFlow Serving repo:
git clone https://github.com/tensorflow/serving.git
export TF_SERVING_ROOT=$(pwd)/serving

# Setup a virtual environment:
python3 -m virtualenv -p python3 .venv3
. .venv3/bin/activate
pip install requests

# Send rRPC request to your served model:
python $TF_SERVING_ROOT/tensorflow_serving/example/resnet_client.py
```
This will return something like the following:
```shell
Prediction class: 286, avg latency: 57.318000000000005 ms
```
