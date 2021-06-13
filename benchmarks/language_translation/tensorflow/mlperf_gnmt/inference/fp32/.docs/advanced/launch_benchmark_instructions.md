<!-- 50. Launch benchmark instructions -->
If you are going to run using docker, copy the `tensorflow-addons` wheel that you built
during the [model setup](README.md#run-the-model) to the model zoo's `mlperf_gnmt` directory:
```
cp <tensorflow-addons repo>/artifacts/tensorflow_addons-*.whl <model zoo directory>/models/language_translation/tensorflow/mlperf_gnmt
```

Once your environment is setup, navigate to the `benchmarks` directory of
the model zoo and set environment variables for the dataset, checkpoint
directory, and an output directory where log files will be written.
```
cd benchmarks

export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
export PRETRAINED_MODEL=<path to the pretrained model frozen graph .pb file>
```

<model name> <mode> can be run in three different modes:

* For online inference, use the following command (with `--benchmark-only`, `--socket-id 0` and `--batch-size 1`):
  ```
  python launch_benchmark.py \
    --model-name mlperf_gnmt \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --batch-size 1 \
    --socket-id 0 \
    --data-location $DATASET_DIR \
    --docker-image <docker image> \
    --in-graph $PRETRAINED_MODEL \
    --output-dir $OUTPUT_DIR \
    --benchmark-only
  ```
* For batch inference, use the following command (with `--benchmark-only`, `--socket-id 0` and `--batch-size 32`):
  ```
  python launch_benchmark.py \
    --model-name mlperf_gnmt \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --batch-size 32 \
    --socket-id 0 \
    --data-location $DATASET_DIR \
    --docker-image <docker image> \
    --in-graph $PRETRAINED_MODEL \
    --output-dir $OUTPUT_DIR \
    --benchmark-only
  ```
* For accuracy testing, use the following command (with `--accuracy_only`, `--socket-id 0` and `--batch-size 32`):
  ```
  python launch_benchmark.py \
    --model-name mlperf_gnmt \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --batch-size 32 \
    --socket-id 0 \
    --data-location $DATASET_DIR \
    --docker-image <docker image> \
    --in-graph $PRETRAINED_MODEL \
    --output-dir $OUTPUT_DIR \
    --accuracy-only
  ```
