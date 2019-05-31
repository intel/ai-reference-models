# FaceNet

This document has instructions for how to run FaceNet for the
following modes/precisions:
* [FP32 inference](#fp32-inference-instructions)

Script instructions for model training and inference for other precisions are coming later.

## FP32 Inference Instructions

1. Clone the [davidsandberg/facenet](https://github.com/davidsandberg/facenet) repository:

```
$ git clone https://github.com/davidsandberg/facenet.git
```

2. Clone this [intelai/models](https://github.com/IntelAI/models)
repository:

```
$ git clone https://github.com/IntelAI/models.git
```

3. Download and extract the pre-trained model.
```
$ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/facenet_fp32_pretrained_model.tar.gz
$ tar -zxvf facenet_fp32_pretrained_model.tar.gz
$ ls checkpoint
model-20181015-153825.ckpt-90.data-00000-of-00001  model-20181015-153825.ckpt-90.index  model-20181015-153825.meta
```

4. If you would like to run FaceNet FP32 inference, you will need the aligned LFW dataset.
Instructions for downloading the dataset and converting it can be found in the documentation
[here](https://github.com/davidsandberg/facenet/wiki/Validate-on-lfw), step 2 to step 4.

5. Navigate to the `benchmarks` directory in your local clone of
the [intelai/models](https://github.com/IntelAI/models) repo from step 2.
The `launch_benchmark.py` script in the `benchmarks` directory is
used for starting a model run in a optimized TensorFlow docker
container. It has arguments to specify which model, framework, mode,
precision, and docker image.

Substitute in your own `--checkpoint` pretrained model file path (from step 3),
and `--data-location` (from step 4).

FaceNet can be run for testing online inference, batch inference, or accuracy. 
Use one of the following examples below, depending on your use case.

* For online inference (using `--batch-size 1`):

```
python launch_benchmark.py \
    --model-name facenet \
    --precision fp32 \
    --mode inference \
    --framework tensorflow \
    --batch-size 1 \
    --socket-id 0 \
    --checkpoint /home/<user>/checkpoints \
    --data-location  /home/<user>/dataset \
    --model-source-dir /home/<user>/facenet/ \
    --docker-image intelaipg/intel-optimized-tensorflow:1.14
```
Example log tail for online inference:
```
Batch 979 elapsed Time 0.0297989845276
Batch 989 elapsed Time 0.029657125473
Batch 999 elapsed Time 0.0295519828796
Batchsize: 1
Time spent per BATCH: 30.1561 ms
Total samples/sec: 33.1608 samples/s
2019-03-28 21:00:02.725536: W tensorflow/core/kernels/queue_base.cc:277] _2_input_producer: Skipping cancelled enqueue attempt with queue not closed
2019-03-28 21:00:02.725672: W tensorflow/core/kernels/queue_base.cc:277] _1_batch_join/fifo_queue: Skipping cancelled enqueue attempt with queue not closed
2019-03-28 21:00:02.725683: W tensorflow/core/kernels/queue_base.cc:277] _1_batch_join/fifo_queue: Skipping cancelled enqueue attempt with queue not closed
2019-03-28 21:00:02.725693: W tensorflow/core/kernels/queue_base.cc:277] _1_batch_join/fifo_queue: Skipping cancelled enqueue attempt with queue not closed
2019-03-28 21:00:02.725713: W tensorflow/core/kernels/queue_base.cc:277] _1_batch_join/fifo_queue: Skipping cancelled enqueue attempt with queue not closed
2019-03-28 21:00:02.725722: W tensorflow/core/kernels/queue_base.cc:277] _1_batch_join/fifo_queue: Skipping cancelled enqueue attempt with queue not closed
2019-03-28 21:00:02.725746: W tensorflow/core/kernels/queue_base.cc:277] _1_batch_join/fifo_queue: Skipping cancelled enqueue attempt with queue not closed
2019-03-28 21:00:02.725776: W tensorflow/core/kernels/queue_base.cc:277] _1_batch_join/fifo_queue: Skipping cancelled enqueue attempt with queue not closed
Ran inference with batch size 1
Log location outside container: {--output-dir value}/benchmark_facenet_inference_fp32_20190328_205911.log
```

* For batch inference (using `--batch-size 100`):

```
python launch_benchmark.py \
    --model-name facenet \
    --precision fp32 \
    --mode inference \
    --framework tensorflow \
    --batch-size 100 \
    --socket-id 0 \
    --checkpoint /home/<user>/checkpoints \
    --data-location  /home/<user>/dataset \
    --model-source-dir /home/<user>/facenet/ \
    --docker-image intelaipg/intel-optimized-tensorflow:1.14
```
Example log tail for batch inference:
```
Batch 219 elapsed Time 0.446497917175
Batch 229 elapsed Time 0.422048091888
Batch 239 elapsed Time 0.433968067169
Batchsize: 100
Time spent per BATCH: 434.5414 ms
Total samples/sec: 230.1277 samples/s
Accuracy: 0.98833+-0.00489
Validation rate: 0.96200+-0.01968 @ FAR=0.00100
Area Under Curve (AUC): 0.999
Equal Error Rate (EER): 0.011
Ran inference with batch size 100
Log location outside container: {--output-dir value}/benchmark_facenet_inference_fp32_20190329_002623.log
```

* For accuracy (using `--accuracy-only`, and `--batch-size 100`):

```
python launch_benchmark.py \
    --model-name facenet \
    --precision fp32 \
    --mode inference \
    --framework tensorflow \
    --accuracy-only \
    --batch-size 100 \
    --socket-id 0 \
    --checkpoint /home/<user>/checkpoints \
    --data-location  /home/<user>/dataset \
    --model-source-dir /home/<user>/facenet/ \
    --docker-image intelaipg/intel-optimized-tensorflow:1.14
```
Example log tail for accuracy:
```
Batch 219 elapsed Time 0.398629188538
Batch 229 elapsed Time 0.354953050613
Batch 239 elapsed Time 0.366483926773
Batchsize: 100
Time spent per BATCH: 388.5419 ms
Total samples/sec: 257.3725 samples/s
Accuracy: 0.98833+-0.00489
Validation rate: 0.96200+-0.01968 @ FAR=0.00100
Area Under Curve (AUC): 0.999
Equal Error Rate (EER): 0.011
Ran inference with batch size 100
Log location outside container: {--output-dir value}/benchmark_facenet_inference_fp32_20190328_214145.log
```

Note that the `--verbose` or `--output-dir` flag can be added to any of the above commands
to get additional debug output or change the default output location..