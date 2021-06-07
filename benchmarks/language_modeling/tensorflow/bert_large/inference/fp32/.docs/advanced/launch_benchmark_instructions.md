<!-- 50. Launch benchmark instructions -->
Once your environment is setup, navigate to the `benchmarks` directory of
the model zoo and set environment variables for the dataset, checkpoint
directory, frozen graph, and an output directory where log files will be written.
```
cd benchmarks

export DATASET_DIR=<path to the dataset>
export CHECKPOINT_DIR=<path to the pretrained model checkpoints>
export PRETRAINED_MODEL=<path to the frozen graph .pb file>
export OUTPUT_DIR=<directory where log files will be saved>
```

<model name> <mode> can be run in three different modes:

* Benchmark
  ```
  python launch_benchmark.py \
    --model-name=bert_large \
    --precision=fp32 \
    --mode=inference \
    --framework=tensorflow \
    --batch-size=128 \
    --data-location $DATASET_DIR \
    --checkpoint $CHECKPOINT_DIR \
    --in-graph $PRETRAINED_MODEL \
    --output-dir $OUTPUT_DIR \
    --docker-image <docker image> \
    --benchmark-only \
    -- infer_option=SQuAD
  ```
* Profile
  ```
  python launch_benchmark.py \
    --model-name=bert_large \
    --precision=fp32 \
    --mode=inference \
    --framework=tensorflow \
    --batch-size=32 \
    --data-location $DATASET_DIR \
    --checkpoint $CHECKPOINT_DIR \
    --in-graph $PRETRAINED_MODEL \
    --docker-image <docker image> \
    --output-dir $OUTPUT_DIR \
    -- profile=True infer_option=SQuAD
  ```
* Accuracy
  ```
  python launch_benchmark.py \
    --model-name=bert_large \
    --precision=fp32 \
    --mode=inference \
    --framework=tensorflow \
    --batch-size=32 \
    --data-location $DATASET_DIR \
    --checkpoint $CHECKPOINT_DIR \
    --in-graph $PRETRAINED_MODEL \
    --docker-image <docker image> \
    --output-dir $OUTPUT_DIR \
    --accuracy-only \
    -- infer_option=SQuAD
  ```

Output files and logs are saved to the ${OUTPUT_DIR} directory.
