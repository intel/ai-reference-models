<!-- 50. Launch benchmark instructions -->
Once your environment is setup, navigate to the `benchmarks` directory of
the model zoo and set environment variables pointing to the directory for the
dataset, pretrained model frozen graph, and an output directory where log
files will be written.

```
# cd to the benchmarks directory in the model zoo
cd benchmarks

export DATASET_DIR=<path to the dataset (only required for accuracy)>
export OUTPUT_DIR=<directory where log files will be written>
export PRETRAINED_MODEL=<path to the frozen graph that you downloaded>
```

Inception V4 can be run to test accuracy, batch inference, or online inference.
Use one of the following examples below, depending on your use case.

* For accuracy run the following command that uses the `DATASET_DIR`, a batch
  size of 100, and the `--accuracy-only` flag:
  ```
  python launch_benchmark.py \
    --model-name inceptionv4 \
    --precision fp32 \
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

 * For batch inference, use the command below that has a batch size of 240
   and the `--benchmark-only` flag. Synthetic data will be used, since no
   dataset is being provided.
   ```
   python launch_benchmark.py \
     --model-name inceptionv4 \
     --precision fp32 \
     --mode inference \
     --framework tensorflow \
     --benchmark-only \
     --batch-size 240 \
     --socket-id 0 \
     --docker-image <docker image> \
     --in-graph ${PRETRAINED_MODEL} \
     --output-dir ${OUTPUT_DIR}
   ```

* For online inference use the command below that has a batch size of 1 and
  the `--benchmark-only` flag. Again, synthetic data is being used in this
  example.
  ```
  python launch_benchmark.py \
    --model-name inceptionv4 \
    --precision fp32 \
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
Iteration time: 1337.8728 ms
Processed 49600 images. (Top1 accuracy, Top5 accuracy) = (0.8015, 0.9517)
Iteration time: 1331.8253 ms
Processed 49700 images. (Top1 accuracy, Top5 accuracy) = (0.8017, 0.9518)
Iteration time: 1339.1553 ms
Processed 49800 images. (Top1 accuracy, Top5 accuracy) = (0.8017, 0.9518)
Iteration time: 1334.5991 ms
Processed 49900 images. (Top1 accuracy, Top5 accuracy) = (0.8018, 0.9519)
Iteration time: 1336.1905 ms
Processed 50000 images. (Top1 accuracy, Top5 accuracy) = (0.8018, 0.9519)
Ran inference with batch size 100
Log location outside container: ${OUTPUT_DIR}/benchmark_inceptionv4_inference_fp32_20190308_182729.log
```

Example log tail when running for batch inference:
```
[Running warmup steps...]
steps = 10, 91.4372832625 images/sec
[Running benchmark steps...]
steps = 10, 91.0217283977 images/sec
steps = 20, 90.8331507586 images/sec
steps = 30, 91.1284943026 images/sec
steps = 40, 91.1885998597 images/sec
steps = 50, 91.1905741783 images/sec
Ran inference with batch size 240
Log location outside container: ${OUTPUT_DIR}/benchmark_inceptionv4_inference_fp32_20190308_184431.log
```

Example log tail when running for online inference:
```
[Running warmup steps...]
steps = 10, 15.6993019295 images/sec
[Running benchmark steps...]
steps = 10, 16.3553780883 images/sec
steps = 20, 15.771143231 images/sec
steps = 30, 15.7133587586 images/sec
steps = 40, 16.0477494988 images/sec
steps = 50, 15.483992912 images/sec
Latency: 63.534 ms
Ran inference with batch size 1
Log location outside container: ${OUTPUT_DIR}/benchmark_inceptionv4_inference_fp32_20190307_221954.log
```
