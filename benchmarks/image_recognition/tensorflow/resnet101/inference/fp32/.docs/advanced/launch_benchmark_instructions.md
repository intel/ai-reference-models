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
  --model-name resnet101 \
  --precision fp32 \
  --mode inference \
  --framework tensorflow \
  --in-graph ${PRETRAINED_MODEL} \
  --data-location ${DATASET_DIR} \
  --output-dir ${OUTPUT_DIR} \
  --batch-size 128 \
  --socket-id 0 \
  --docker-image <docker image>
```

* For online inference, use the command below that uses the `DATASET_DIR` and a batch 
  size of 1.
  
```
python launch_benchmark.py \
  --model-name resnet101 \
  --precision fp32 \
  --mode inference \
  --framework tensorflow \
  --in-graph ${PRETRAINED_MODEL} \
  --data-location ${DATASET_DIR} \
  --output-dir ${OUTPUT_DIR} \
  --batch-size 1 \
  --socket-id 0 \
  --docker-image <docker image>
```

Example log file snippet when testing accuracy:
```
Processed 49600 images. (Top1 accuracy, Top5 accuracy) = (0.7639, 0.9289)
Processed 49700 images. (Top1 accuracy, Top5 accuracy) = (0.7639, 0.9290)
Processed 49800 images. (Top1 accuracy, Top5 accuracy) = (0.7639, 0.9289)
Processed 49900 images. (Top1 accuracy, Top5 accuracy) = (0.7641, 0.9289)
Processed 50000 images. (Top1 accuracy, Top5 accuracy) = (0.7640, 0.9289)
Ran inference with batch size 100
Log location outside container: {--output-dir value}/benchmark_resnet101_inference_fp32_20190104_201506.log
```

Example log file snippet when testing batch or online inference:
```
steps = 70, ... images/sec
steps = 80, ... images/sec
steps = 90, ... images/sec
steps = 100, ... images/sec
Ran inference with batch size 128
Log location outside container: {--output-dir value}/benchmark_resnet101_inference_fp32_20190104_204615.log
```
