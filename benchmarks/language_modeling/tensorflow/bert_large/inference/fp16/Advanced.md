<!--- 0. Title -->
<!-- This document is auto-generated using markdown fragments and the model-builder -->
<!-- To make changes to this doc, please change the fragments instead of modifying this doc directly -->
# BERT Large FP16 inference - Advanced Instructions

<!-- 10. Description -->
This document has advanced instructions for running BERT Large FP16
inference, which provides more control over the individual parameters that
are used. For more information on using [`/benchmarks/launch_benchmark.py`](/benchmarks/launch_benchmark.py),
see the [launch benchmark documentation](/docs/general/tensorflow/LaunchBenchmark.md).

Prior to using these instructions, please follow the setup instructions from
the model's [README](README.md) and/or the
[AI Kit documentation](/docs/general/tensorflow/AIKit.md) to get your environment
setup (if running on bare metal) and download the dataset, pretrained model, etc.
If you are using AI Kit, please exclude the `--docker-image` flag from the
commands below, since you will be running the the TensorFlow conda environment
instead of docker.

<!-- 55. Docker arg -->
Any of the `launch_benchmark.py` commands below can be run on bare metal by
removing the `--docker-image` arg. Ensure that you have all of the
[required prerequisites installed](README.md#run-the-model) in your environment
before running without the docker container.

If you are new to docker and are running into issues with the container,
see [this document](/docs/general/docker.md) for troubleshooting tips.

<!-- 50. Launch benchmark instructions -->
Once your environment is setup, navigate to the `benchmarks` directory of
the model zoo and set environment variables for the dataset, checkpoint
directory, and an output directory where log files will be written.
```
cd benchmarks

export DATASET_DIR=<path to the dataset>
export CHECKPOINT_DIR=<path to the pretrained model checkpoints>
export OUTPUT_DIR=<directory where log files will be saved>
```

BERT Large inference can be run in three different modes:

* Benchmark
  ```
  python launch_benchmark.py \
    --model-name=bert_large \
    --precision=fp16 \
    --mode=inference \
    --framework=tensorflow \
    --batch-size=32 \
    --data-location ${DATASET_DIR} \
    --checkpoint ${CHECKPOINT_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --docker-image intel/intel-optimized-tensorflow:latest \
    --benchmark-only \
    -- infer_option=SQuAD
  ```
* Profile
  ```
  python launch_benchmark.py \
    --model-name=bert_large \
    --precision=fp16 \
    --mode=inference \
    --framework=tensorflow \
    --batch-size=32 \
    --data-location ${DATASET_DIR}  \
    --checkpoint ${CHECKPOINT_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --docker-image intel/intel-optimized-tensorflow:latest \
    -- profile=True infer_option=SQuAD
  ```
* Accuracy
  ```
  python launch_benchmark.py \
    --model-name=bert_large \
    --precision=fp16 \
    --mode=inference \
    --framework=tensorflow \
    --batch-size=32 \
    --data-location ${DATASET_DIR} \
    --checkpoint ${CHECKPOINT_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --docker-image intel/intel-optimized-tensorflow:latest \
    --accuracy-only \
    -- infer_option=SQuAD
  ```

Output files and logs are saved to the ${OUTPUT_DIR} directory.

<!-- 70. Model args -->
Note that args specific to this model are specified after ` -- ` at
the end of the command (like the `profile=True` arg in the Profile
command above. Below is a list of all of the model specific args and
their default values:

| Model arg | Default value |
|-----------|---------------|
| doc_stride | `128` |
| max_seq_length | `384` |
| profile | `False` |
| config_file | `bert_config.json` |
| vocab_file | `vocab.txt` |
| predict_file | `dev-v1.1.json` |
| init_checkpoint | `model.ckpt-3649` |

