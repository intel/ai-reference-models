<!-- 50. Launch benchmark instructions -->
Once your environment is setup, navigate to the `benchmarks` directory of
the model zoo and set environment variables pointing to the directory for the
dataset, pretrained model frozen graph, and an output directory where log
files will be written.

```
# cd to the benchmarks directory in the model zoo
cd benchmarks

export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
export PRETRAINED_MODEL=<path to the frozen graph that you downloaded>
```

Inception V4 can be run to test accuracy, batch inference, or online inference.
Use one of the following examples below, depending on your use case.

* Use the command below to test accuracy. This command uses the `DATASET_DIR`,
  a batch size of 100 and the `--accuracy-only` flag:
  ```
  python launch_benchmark.py \
    --model-name inceptionv4 \
    --precision int8 \
    --mode inference \
    --framework tensorflow \
    --accuracy-only \
    --batch-size 100 \
    --socket-id 0 \
    --docker-image <docker image> \
    --in-graph ${PRETRAINED_MODEL} \
    --data-location ${DATASET_DIR} \
    --output-dir ${OUTPUT_DIR}
  ```

* For batch inference, use the command below that uses a batch size of 240 and
  the `--benchmark-only` flag. Since no dataset is being provided, synthetic data
  will be used.
  ```
  python launch_benchmark.py \
    --model-name inceptionv4 \
    --precision int8 \
    --mode inference \
    --framework tensorflow \
    --benchmark-only \
    --batch-size 240 \
    --socket-id 0 \
    --docker-image <docker image> \
    --in-graph ${PRETRAINED_MODEL} \
    --output-dir ${OUTPUT_DIR}
  ```

* For online inference, use the command below that has a batch size of 1 and the
  `--benchmark-only` flag. Again, synthetic data is being used in this example.
  ```
  python launch_benchmark.py \
    --model-name inceptionv4 \
    --precision int8 \
    --mode inference \
    --framework tensorflow \
    --benchmark-only \
    --batch-size 1 \
    --socket-id 0 \
    --docker-image <docker image> \
    --in-graph ${PRETRAINED_MODEL} \
    --output-dir ${OUTPUT_DIR}
  ```

Note that the `--verbose` flag can be added to any of the above commands
to get additional debug output.

The log file is saved to the `OUTPUT_DIR` directory. Below are examples of
what the tail of your log file should look like for the different configs.

Example log tail when running for accuracy:
```
...
Iteration time: 685.1976 ms
Processed 49700 images. (Top1 accuracy, Top5 accuracy) = (0.7985, 0.9504)
Iteration time: 686.3845 ms
Processed 49800 images. (Top1 accuracy, Top5 accuracy) = (0.7983, 0.9504)
Iteration time: 686.7021 ms
Processed 49900 images. (Top1 accuracy, Top5 accuracy) = (0.7984, 0.9504)
Iteration time: 685.8914 ms
Processed 50000 images. (Top1 accuracy, Top5 accuracy) = (0.7984, 0.9504)
Ran inference with batch size 100
Log location outside container: ${OUTPUT_DIR}/benchmark_inceptionv4_inference_int8_20190306_221608.log
```

Example log tail when running for batch inference:
```
[Running warmup steps...]
steps = 10, 184.497605972 images/sec
[Running benchmark steps...]
steps = 10, 184.664702184 images/sec
steps = 20, 184.938455688 images/sec
steps = 30, 184.454197634 images/sec
steps = 40, 184.491891402 images/sec
steps = 50, 184.390001575 images/sec
Ran inference with batch size 240
Log location outside container: ${OUTPUT_DIR}/benchmark_inceptionv4_inference_int8_20190415_233517.log
```

Example log tail when running for online inference:
```
[Running warmup steps...]
steps = 10, 32.6095380262 images/sec
[Running benchmark steps...]
steps = 10, 32.9024373024 images/sec
steps = 20, 32.5328989723 images/sec
steps = 30, 32.5988932413 images/sec
steps = 40, 31.3991914957 images/sec
steps = 50, 32.7053998207 images/sec
Latency: 30.598 ms
Ran inference with batch size 1
Log location outside container: ${OUTPUT_DIR}/benchmark_inceptionv4_inference_int8_20190415_232441.log
```
