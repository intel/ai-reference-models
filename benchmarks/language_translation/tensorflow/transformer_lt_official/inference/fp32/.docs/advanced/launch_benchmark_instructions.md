<!-- 50. Launch benchmark instructions -->
Once your environment is setup, navigate to the `benchmarks` directory of
the model zoo and set environment variables for the dataset, checkpoint
directory, and an output directory where log files will be written.
```
cd benchmarks

export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
export PRETRAINED_MODEL=<path to the pretrained model frozen graph .pb file>
```

Transformer LT official can run for online or batch inference. Use one of the
following examples below, depending on your use case.

For online inference (using `--socket-id 0` and `--batch-size 1`):
```
python launch_benchmark.py \
    --model-name transformer_lt_official \
    --precision fp32 \
    --mode inference \
    --framework tensorflow \
    --batch-size 1 \
    --socket-id 0 \
    --docker-image <docker image> \
    --in-graph ${PRETRAINED_MODEL} \
    --data-location ${DATASET_DIR} \
    --output-dir ${OUTPUT_DIR} \
    -- file=newstest2014.en \
    file_out=translate.txt \
    reference=newstest2014.de \
    vocab_file=vocab.txt
```

For batch inference (using `--socket-id 0` and `--batch-size 64`):
```
python launch_benchmark.py \
    --model-name transformer_lt_official \
    --precision fp32 \
    --mode inference \
    --framework tensorflow \
    --batch-size 64 \
    --socket-id 0 \
    --docker-image <docker image> \
    --in-graph ${PRETRAINED_MODEL} \
    --data-location ${DATASET_DIR} \
    --output-dir ${OUTPUT_DIR} \
    -- file=newstest2014.en \
    file_out=translate.txt \
    reference=newstest2014.de \
    vocab_file=vocab.txt

```

Note that the `--verbose` flag can be added to any of the above commands
to get additional debug output.
The num-inter-threads and num-intra-threads could be set different numbers
depending on the CPU in the system to achieve the best performance.
