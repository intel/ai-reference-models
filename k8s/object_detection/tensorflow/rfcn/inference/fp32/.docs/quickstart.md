<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`fp32_inference.sh`](mlops/serving/user-mounted-nfs/pod.yaml#L16) | Runs inference on a directory of raw images for 500 steps and outputs performance metrics. |
| [`fp32_accuracy.sh`](mlops/pipeline/user-mounted-nfs/serving_accuracy.yaml#L49) | Processes the TF records to run inference and check accuracy on the results. |

These quickstart scripts can be run in the following environment:
* [Kubernetes](#kubernetes)

