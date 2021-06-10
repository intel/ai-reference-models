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
  --model-name resnet50 \
  --precision int8 \
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

* For batch inference, use the command below that uses the `DATASET_DIR`, a batch 
  size of 128, 500 steps, and 50 warmup_steps.

```
python launch_benchmark.py \
  --model-name resnet50 \
  --precision int8 \
  --mode inference \
  --framework tensorflow \
  --in-graph ${PRETRAINED_MODEL} \
  --data-location ${DATASET_DIR} \
  --output-dir ${OUTPUT_DIR} \
  --batch-size 128 \
  --benchmark-only \
  --socket-id 0 \
  --docker-image <docker image>
  -- warmup_steps=50 steps=500
```

Example log file snippet when testing accuracy:
```
Iteration time: ... ms
Processed 49600 images. (Top1 accuracy, Top5 accuracy) = (0.7361, 0.9155)
Iteration time: ... ms
Processed 49700 images. (Top1 accuracy, Top5 accuracy) = (0.7361, 0.9155)
Iteration time: ... ms
Processed 49800 images. (Top1 accuracy, Top5 accuracy) = (0.7360, 0.9154)
Iteration time: ... ms
Processed 49900 images. (Top1 accuracy, Top5 accuracy) = (0.7361, 0.9155)
Iteration time: ... ms
Processed 50000 images. (Top1 accuracy, Top5 accuracy) = (0.7360, 0.9154)
Ran inference with batch size 100
Log location outside container: {--output-dir value}/benchmark_resnet50_inference_int8_20190104_212224.log
```

Example log file snippet when testing batch inference:
```
...
Iteration 497: ... sec
Iteration 498: ... sec
Iteration 499: ... sec
Iteration 500: ... sec
Average time: ... sec
Batch size = 128
Throughput: ... images/sec
Ran inference with batch size 128
Log location outside container: {--output-dir value}/benchmark_resnet50_inference_int8_20190416_172735.log
```
