<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`fp32_online_inference.sh`](fp32_online_inference.sh) | Runs online inference (batch_size=1). |
| [`fp32_batch_inference.sh`](fp32_batch_inference.sh) | Runs batch inference (batch_size=128). |
| [`fp32_accuracy.sh`](fp32_accuracy.sh) | Measures the model accuracy (batch_size=100). |
| [`run_tf_serving_client.py`](run_tf_serving_client.py) | Runs gRPC client for multi-node batch and online inference. |
| [`multi_client.sh`](multi_client.sh) | Runs multiple parallel gRPC clients for multi-node batch and online inference. |

These quickstart scripts can be run in different environments:
* [Bare Metal](#bare-metal)
* [Docker](#docker)
* [Kubernetes](#kubernetes)

