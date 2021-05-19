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

The following commonds are examples on how <model name> can be run:
* Running the model for online inference:
  ```
  python launch_benchmark.py \
    --framework tensorflow \
    --model-source-dir $TF_MODELS_DIR \
    --precision fp32 \
    --mode inference \
    --model-name wide_deep \
    --batch-size 1 \
    --data-location $DATASET_DIR \
    --checkpoint $PRETRAINED_MODEL \
    --docker-image <docker image> \
    --output-dir $OUTPUT_DIR \
    --verbose
  ```
  The three locations used (model-source-dir, data-location, checkpoint) here,
  works better with docker if they are located in the local disk. The locations
  should be pointed as absolute path.
* Running the model in batch inference mode:
  ```
  python launch_benchmark.py \
    --framework tensorflow \
    --model-source-dir $TF_MODELS_DIR \
    --precision fp32 \
    --mode inference \
    --model-name wide_deep \
    --batch-size 1024 \
    --data-location $DATASET_DIR \
    --checkpoint $PRETRAINED_MODEL \
    --docker-image <docker image> \
    --output-dir $OUTPUT_DIR \
    --verbose
  ```
  The three locations used (model-source-dir, data-location, checkpoint) here,
  works better with docker if they are located in the local disk. The locations
  should be pointed as absolute path.
