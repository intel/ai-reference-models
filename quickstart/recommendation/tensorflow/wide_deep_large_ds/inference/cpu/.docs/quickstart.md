<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`online_inference.sh`](/quickstart/recommendation/tensorflow/wide_deep_large_ds/inference/cpu/online_inference.sh) | Runs online inference (`batch_size=1`). The `NUM_OMP_THREADS` environment variable and the hyperparameters `num-intra-threads`, `num-inter-threads` can be tuned for best performance. If `NUM_OMP_THREADS` is not set, it will default to `1`. |
| [`accuracy.sh`](/quickstart/recommendation/tensorflow/wide_deep_large_ds/inference/cpu/accuracy.sh) | Measures the model accuracy (`batch_size=1000`). |
