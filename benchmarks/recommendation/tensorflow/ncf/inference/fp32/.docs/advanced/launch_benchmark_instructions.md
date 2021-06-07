<!-- 50. Launch benchmark instructions -->
Once your environment is setup, navigate to the `benchmarks` directory of
the model zoo and set environment variables pointing to the directory for the
dataset, pretrained model, and an output directory where log
files will be written.

```
# cd to the benchmarks directory in the model zoo
cd benchmarks

export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
export PRETRAINED_MODEL=<path to the pretrained model>
export TF_MODELS_DIR=<path to your clone of the TensorFlow models repo>
```

<model name> <precision> <mode> can be run a few different modes:
* For batch inference, `--batch-size 256`, `--socket-id 0`:
  ```
  python launch_benchmark.py \
    --checkpoint $PRETRAINED_MODEL \
    --model-source-dir $TF_MODELS_DIR \
    --data-location ${DATASET_DIR} \
    --model-name ncf \
    --socket-id 0 \
    --batch-size 256 \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --output-dir $OUTPUT_DIR \
    --docker-image <docker image>
  ```
* For online inference, `--batch-size 1`, `--socket-id 0`:
  ```
  python launch_benchmark.py \
    --checkpoint $PRETRAINED_MODEL \
    --model-source-dir $TF_MODELS_DIR \
    --data-location ${DATASET_DIR} \
    --model-name ncf \
    --socket-id 0 \
    --batch-size 1 \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --output-dir $OUTPUT_DIR \
    --docker-image <docker image>
  ```
* For Accuracy, `--batch-size 256`, `--socket-id 0`:
  ```
  python launch_benchmark.py \
    --checkpoint $PRETRAINED_MODEL \
    --model-source-dir $TF_MODELS_DIR \
    --data-location ${DATASET_DIR} \
    --model-name ncf \
    --socket-id 0 \
    --accuracy-only \
    --batch-size 256 \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --output-dir $OUTPUT_DIR \
    --docker-image <docker image>
  ```
