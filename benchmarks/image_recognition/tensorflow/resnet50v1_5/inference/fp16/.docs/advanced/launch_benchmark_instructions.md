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

<model name> <precision> <mode> can be run to test accuracy, batch inference, or online inference.
Use one of the following examples below, depending on your use case.

* For accuracy run the following command that uses the `DATASET_DIR`, a batch
  size of 100, and the `--accuracy-only` flag:

```
python launch_benchmark.py \
  --data-location ${DATASET_DIR} \
  --in-graph ${PRETRAINED_MODEL} \
  --model-name resnet50v1_5 \
  --framework tensorflow \
  --precision fp16 \
  --mode inference \
  --batch-size=100 \
  --output-dir ${OUTPUT_DIR} \
  --accuracy-only \
  --docker-image <docker image>
```

* For batch inference, use the command below that uses the `DATASET_DIR`, a batch 
  size of 128, and the `--benchmark-only` flag:

```
python launch_benchmark.py \
  --in-graph ${PRETRAINED_MODEL} \
  --model-name resnet50v1_5 \
  --framework tensorflow \
  --precision fp16 \
  --mode inference \
  --batch-size=128 \
  --output-dir ${OUTPUT_DIR} \
  --data-location ${DATASET_DIR} \
  --benchmark-only \
  --docker-image <docker image>
```

* For online inference, use the command below that uses the `DATASET_DIR`, a batch 
  size of 1, and the `--benchmark-only` flag:
  
```
python launch_benchmark.py \
  --in-graph ${PRETRAINED_MODEL} \
  --model-name resnet50v1_5 \
  --framework tensorflow \
  --precision fp16 \
  --mode inference \
  --batch-size=1 \
  --output-dir ${OUTPUT_DIR} \
  --data-location ${DATASET_DIR} \
  --benchmark-only \
  --docker-image <docker image>
```

Example log file snippet when testing accuracy:
```
...
Iteration time: ... ms
Processed 49900 images. (Top1 accuracy, Top5 accuracy) = (0.7672, 0.9314)
Iteration time: ... ms
Processed 50000 images. (Top1 accuracy, Top5 accuracy) = (0.7672, 0.9314)
Ran inference with batch size 100
Log file location: {--output-dir value}/benchmark_resnet50v1_5_inference_fp16_{timestamp}.log
```

Example log file snippet when testing batch inference:
```
...
Iteration 48: ... sec
Iteration 49: ... sec
Iteration 50: ... sec
Average time: ... sec
Batch size = 128
Throughput: ... images/sec
Ran inference with batch size 128
Log file location: {--output-dir value}/benchmark_resnet50v1_5_inference_fp16_{timestamp}.log
```

Example log file snippet when testing online inference:
```
...
Iteration 48: ... sec
Iteration 49: ... sec
Iteration 50: ... sec
Average time: ... sec
Batch size = 1
Latency: ... ms
Throughput: ... images/sec
Ran inference with batch size 1
Log file location: {--output-dir value}/benchmark_resnet50v1_5_inference_fp16_{timestamp}.log
```

Batch and online inference can also be run with multiple instances using
`numactl`. The following commands have examples how to do multi-instance runs
using the `--numa-cores-per-instance` argument. Note that these examples are
running with real data (specified by `--data-location ${DATASET_DIR}`).
To use synthetic data, you can omit that argument.

* For multi-instance batch inference, the recommended configuration uses all
  the cores on a socket for each instance (this means that if you have 2 sockets,
  you would be running 2 instances - one per socket) and a batch size of 128.
  ```  
  python launch_benchmark.py \
    --model-name resnet50v1_5 \
    --precision fp16 \
    --mode inference \
    --framework tensorflow \
    --in-graph ${PRETRAINED_MODEL} \
    --data-location ${DATASET_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --batch-size 128 \
    --numa-cores-per-instance socket \
    --docker-image <docker image> \
    -- warmup_steps=50 steps=1500
  ```
  
* For multi-instance online inference, the recommended configuration is using
  4 cores per instance and a batch size of 1.
  ```
  python launch_benchmark.py \
    --model-name resnet50v1_5 \
    --precision fp16 \
    --mode inference \
    --framework tensorflow \
    --in-graph ${PRETRAINED_MODEL} \
    --data-location ${DATASET_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --batch-size 1 \
    --numa-cores-per-instance 4 \
    --docker-image <docker image> \
    -- warmup_steps=50 steps=1500
  ```
