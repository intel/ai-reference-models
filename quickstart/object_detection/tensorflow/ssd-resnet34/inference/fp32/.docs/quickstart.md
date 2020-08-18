<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [fp32_accuracy.sh](fp32_accuracy.sh) | Runs an accuracy test using data in the TF records format with an input size of 300x300. |
| [fp32_accuracy_1200.sh](fp32_accuracy_1200.sh) | Runs an accuracy test using data in the TF records format with an input size of 1200x1200. |
| [fp32_inference.sh](fp32_inference.sh) | Runs inference with a batch size of 1 using synthetic data with an input size of 300x300. Prints out the time spent per batch and total samples/second. |
| [fp32_inference_1200.sh](fp32_inference_1200.sh) | Runs inference with a batch size of 1 using synthetic data with an input size of 1200x1200. Prints out the time spent per batch and total samples/second. |


These quickstart scripts can be run in different environments:
* [Bare Metal](#bare-metal)
* [Docker](#docker)
