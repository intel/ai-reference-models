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
  --model-name resnet101 \
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

* When running for performance, it is optional to specify the
  number of `warmup_steps` and `steps` as extra args, as shown in the
  commands below. If these values are not specified, the script will
  default to use `warmup_steps=40` and `steps=100`. For batch inference, 
  use the command below that uses the `DATASET_DIR`, a batch size of 128,
  `warmup_steps=50`, and `steps=500`.

```
python launch_benchmark.py \
  --model-name resnet101 \
  --precision int8 \
  --mode inference \
  --framework tensorflow \
  --in-graph ${PRETRAINED_MODEL} \
  --data-location ${DATASET_DIR} \
  --output-dir ${OUTPUT_DIR} \
  --batch-size 128 \
  --socket-id 0 \
  --docker-image <docker image> \
  -- warmup_steps=50 steps=500
```

* For online inference, use the command below that uses the `DATASET_DIR`, a batch 
  size of 1, `warmup_steps=50`, and `steps=500`.
  
```
python launch_benchmark.py \
  --model-name resnet101 \
  --precision int8 \
  --mode inference \
  --framework tensorflow \
  --in-graph ${PRETRAINED_MODEL} \
  --data-location ${DATASET_DIR} \
  --output-dir ${OUTPUT_DIR} \
  --batch-size 1 \
  --docker-image <docker image> \
  --socket-id 0 \
  -- warmup_steps=50 steps=500
```

Example log file snippet when testing accuracy:
```
Processed 49800 images. (Top1 accuracy, Top5 accuracy) = (0.7690, 0.9304)
Processed 49900 images. (Top1 accuracy, Top5 accuracy) = (0.7691, 0.9305)
Processed 50000 images. (Top1 accuracy, Top5 accuracy) = (0.7691, 0.9305)
Ran inference with batch size 100
Log location outside container: {--output-dir value}/benchmark_resnet101_inference_int8_20190104_205838.log
```

Example log file snippet when testing batch inference:
```
...
steps = 470, ... images/sec
steps = 480, ... images/sec
steps = 490, ... images/sec
steps = 500, ... images/sec
Ran inference with batch size 128
Log location outside container: {--output-dir value}/benchmark_resnet101_inference_int8_20190223_192438.log
```

Example log file snippet when testing online inference:
```
...
steps = 470, ... images/sec
steps = 480, ... images/sec
steps = 490, ... images/sec
steps = 500, ... images/sec
Ran inference with batch size 1
Log location outside container: {--output-dir value}/benchmark_resnet101_inference_int8_20190223_191406.log
```
