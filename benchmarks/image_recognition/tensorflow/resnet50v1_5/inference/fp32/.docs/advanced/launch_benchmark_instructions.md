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
  --model-name resnet50v1_5 \
  --precision fp32 \
  --mode inference \
  --framework tensorflow \
  --in-graph ${PRETRAINED_MODEL} \
  --data-location ${DATASET_DIR} \
  --output-dir ${OUTPUT_DIR} \
  --accuracy-only \
  --batch-size 100 \
  --socket-id 0 \
  --docker-image <docker image>
```

* For batch inference, use the command below that uses the `DATASET_DIR` and a batch 
  size of 128.

```
python launch_benchmark.py \
  --model-name resnet50v1_5 \
  --precision fp32 \
  --mode inference \
  --framework tensorflow \
  --in-graph ${PRETRAINED_MODEL} \
  --data-location ${DATASET_DIR} \
  --output-dir ${OUTPUT_DIR} \
  --batch-size=128 \
  --socket-id 0 \
  --docker-image <docker image> \
  -- warmup_steps=50 steps=1500
```

* For online inference, use the command below that uses the `DATASET_DIR` and a batch 
  size of 1.
  
```
python launch_benchmark.py \
  --model-name=resnet50v1_5 \
  --precision=fp32 \
  --mode=inference \
  --framework tensorflow \
  --in-graph ${PRETRAINED_MODEL} \
  --data-location ${DATASET_DIR} \
  --output-dir ${OUTPUT_DIR} \
  --batch-size=1 \
  --socket-id 0 \
  --docker-image <docker image> \
  -- warmup_steps=50 steps=1500
```

Example log file snippet when testing accuracy:
```
...
Iteration time: ... ms
Processed 50000 images. (Top1 accuracy, Top5 accuracy) = (0.7648, 0.9308)
Ran inference with batch size 100
Log file location: {--output-dir value}/benchmark_resnet50v1_5_inference_fp32_{timestamp}.log
```

Example log file snippet when testing batch inference:
```
Iteration 1: ... sec
Iteration 2: ... sec
Iteration 3: ... sec
...
Iteration 388: ... sec
Iteration 389: ... sec
Iteration 390: ... sec
Average time: ... sec
Batch size = 128
Throughput: ... images/sec
Ran inference with batch size 128
Log file location: {--output-dir value}/benchmark_resnet50v1_5_inference_fp32_{timestamp}.log
```

Example log file snippet when testing online inference:
```
Iteration 1: ... sec
Iteration 2: ... sec
Iteration 3: ... sec
...
Iteration 1498: ... sec
Iteration 1499: ... sec
Iteration 1500: ... sec
Average time: ... sec
Batch size = 1
Latency: ... ms
Throughput: ... images/sec
Ran inference with batch size 1
Log file location: {--output-dir value}/benchmark_resnet50v1_5_inference_fp32_{timestamp}.log
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
    --precision fp32 \
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
    --precision fp32 \
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
