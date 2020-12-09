<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`int8_online_inference.sh`](int8_online_inference.sh) | Runs online inference (`batch_size=1`). The `NUM_OMP_THREADS` environment variable and the hyperparameters `num-intra-threads`, `num-inter-threads` can be tuned for best performance. |
| [`int8_accuracy.sh`](int8_accuracy.sh) | Measures the model accuracy (`batch_size=1000`). |

These quickstart scripts can be run in different environments:
* [Bare Metal](#bare-metal)
* [Docker](#docker)
