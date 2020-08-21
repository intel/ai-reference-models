# R-FCN (ResNet101)

This document has instructions for how to run R-FCN for the
following FP32 and Int8 modes/precisions:
* [Int8 inference](#int8-inference-instructions)
* [FP32 inference](#fp32-inference-instructions)

Instructions and scripts for the R-FCN ResNet101 model training and inference
for other precisions are coming later.

## Int8 Inference Instructions

These instructions use the TCMalloc memory allocator, which produces 
better performance results for Int8 precision models with smaller batch sizes. 
If you want to disable the use of TCMalloc, set `--disable-tcmalloc=True` 
when calling `launch_benchmark.py` and the script will run without TCMalloc.

1. Clone [intelai/models](https://github.com/intelai/models), [tensorflow/models](https://github.com/tensorflow/models) as `tensorflow-models`, and [cocodataset/cocoapi](https://github.com/cocodataset/cocoapi) repositories:

```
$ git clone https://github.com/IntelAI/models.git intel-models
$ git clone https://github.com/tensorflow/models.git tensorflow-models
$ cd tensorflow-models
$ git checkout 6c21084503b27a9ab118e1db25f79957d5ef540b
$ git apply ../intel-models/models/object_detection/tensorflow/rfcn/inference/tf-2.0.patch
$ git clone https://github.com/cocodataset/cocoapi.git

```

The TensorFlow models repo will be used for installing dependencies and running inference as well as
converting the coco dataset to the TF records format.

2. Download and preprocess the COCO validation images using the [instructions here](datasets/coco/README.md).
   Be sure to export the $DATASET_DIR and $OUTPUT_DIR environment variables.

The `coco_val.record` file is what we will use in this inference example.

3. Download the pre-trained model (Int8 graph):

```
$ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/rfcn_resnet101_int8_coco_pretrained_model.pb
```

4. Go to the Model Zoo benchmarks directory and run the scripts for either batch/online inference performance or accuracy.

```
$ cd /home/<user>/intel-models/benchmarks
```

Run for batch and online inference where the `--data-location`
is the path to the directory with the raw coco validation images and the
`--in-graph` is the Int8 pre-trained graph (from step 3):

```
python launch_benchmark.py \
    --model-name rfcn \
    --mode inference \
    --precision int8 \
    --framework tensorflow \
    --docker-image intel/intel-optimized-tensorflow:2.3.0 \
    --model-source-dir /home/<user>/tensorflow-models \
    --data-location ${DATASET_DIR}/val2017 \
    --in-graph /home/<user>/rfcn_resnet101_int8_coco_pretrained_model.pb \
    --verbose \
    --benchmark-only \
    -- number_of_steps=500
```

Or for accuracy where the `--data-location` is the path the directory
where your `coco_val.record` file is located:
```
python launch_benchmark.py \
    --model-name rfcn \
    --mode inference \
    --precision int8 \
    --framework tensorflow \
    --docker-image intel/intel-optimized-tensorflow:2.3.0 \
    --model-source-dir /home/<user>/tensorflow-models \
    --data-location ${OUTPUT_DIR}/coco_val.record \
    --in-graph /home/<user>/rfcn_resnet101_int8_coco_pretrained_model.pb \
    --accuracy-only \
    -- split="accuracy_message"
```

Note that the `--verbose` or `--output-dir` flag can be added to any of the above commands
to get additional debug output or change the default output location.

5. Log files are located at the value of `--output-dir` (or
`intel-models/benchmarks/common/tensorflow/logs` if no path has been specified):

Below is a sample log file tail when running for batch
and online inference:
```
Step 0: ... seconds
Step 10: ... seconds
...
Step 460: ... seconds
Step 470: ... seconds
Step 480: ... seconds
Step 490: ... seconds
Avg. Duration per Step: ...
...
Ran inference with batch size -1
Log location outside container: {--output-dir}/benchmark_rfcn_inference_int8_20190416_182445.log
```

And here is a sample log file tail when running for accuracy:
```
...
Accumulating evaluation results...
DONE (t=1.91s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.327
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.506
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.365
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.327
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.271
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.376
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.380
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.380
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
Ran inference with batch size -1
Log location outside container: {--output-dir}/benchmark_rfcn_inference_int8_20190227_194752.log
```

## FP32 Inference Instructions

These instructions use the TCMalloc memory allocator, which produces 
better performance results for FP32 precision models with smaller batch sizes. 
If you want to disable the use of TCMalloc, set `--disable-tcmalloc=True` 
when calling `launch_benchmark.py` and the script will run without TCMalloc.

1. Clone [intelai/models](https://github.com/intelai/models), [tensorflow/models](https://github.com/tensorflow/models) as `tensorflow-models`, and [cocodataset/cocoapi](https://github.com/cocodataset/cocoapi) repositories:

```
$ git clone https://github.com/IntelAI/models.git intel-models
$ git clone https://github.com/tensorflow/models.git tensorflow-models
$ cd tensorflow-models
$ git checkout 6c21084503b27a9ab118e1db25f79957d5ef540b
$ git apply ../intel-models/models/object_detection/tensorflow/rfcn/inference/tf-2.0.patch
$ git clone https://github.com/cocodataset/cocoapi.git
```

The TensorFlow models repo will be used for installing dependencies and running inference.

2. Download and preprocess the COCO validation images using the [instructions here](datasets/coco/README.md).
   Be sure to export the $DATASET_DIR and $OUTPUT_DIR environment variables.

3. Download the pre-trained model (FP32 graph):

```
$ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/rfcn_resnet101_fp32_coco_pretrained_model.tar.gz
$ tar -xzvf rfcn_resnet101_fp32_coco_pretrained_model.tar.gz
```

4. Go to the Model Zoo benchmarks directory and run the scripts for either batch/online inference performance or accuracy.

```
$ cd /home/<user>/intel-models/benchmarks
```

Run for batch and online inference where the `--data-location`
is the path to the directory with the raw coco validation images and the
`--in-graph` is the FP32 pre-trained graph (from step 3):

```
python launch_benchmark.py \
    --model-name rfcn \
    --mode inference \
    --precision fp32 \
    --framework tensorflow \
    --docker-image intel/intel-optimized-tensorflow:2.3.0 \
    --model-source-dir /home/<user>/tensorflow-models \
    --data-location ${DATASET_DIR}/val2017 \
    --in-graph /home/<user>/rfcn_resnet101_coco_2018_01_28/frozen_inference_graph.pb \
    --verbose \
    --benchmark-only \
    -- number_of_steps=500
```

Or for accuracy where the `--data-location` is the path the directory
where your `coco_val.record` file is located and the `--in-graph` is
the pre-trained graph located in the pre-trained model directory (from step 3):
```
python launch_benchmark.py \
    --model-name rfcn \
    --mode inference \
    --precision fp32 \
    --framework tensorflow \
    --docker-image intel/intel-optimized-tensorflow:2.3.0 \
    --model-source-dir /home/<user>/tensorflow-models \
    --data-location ${OUTPUT_DIR}/coco_val.record \
    --in-graph /home/<user>/rfcn_resnet101_coco_2018_01_28/frozen_inference_graph.pb \
    --accuracy-only \
    -- split="accuracy_message"
```

Note that the `--verbose` or `--output-dir` flag can be added to any of the above commands
to get additional debug output or change the default output location.

5. Log files are located at the value of `--output-dir` (or
`intel-models/benchmarks/common/tensorflow/logs` if no path has been specified):

Below is a sample log file tail when running for batch
and online inference:
```
Average time per step: ... sec
Received these standard args: Namespace(accuracy_only=False, batch_size=1, benchmark_only=False, checkpoint='/checkpoints', data_location='/dataset', framework='tensorflow', input_graph=None, intelai_models='/workspace/intelai_models', mode='inference', model_args=[], model_name='rfcn', model_source_dir='/workspace/models', num_cores=-1, num_inter_threads=2, num_intra_threads=56, precision='fp32, socket_id=0, use_case='object_detection', verbose=True)
Received these custom args: ['--config_file=rfcn_pipeline.config']
Run model here.
current directory: /workspace/models/research
Running: numactl --cpunodebind=0 --membind=0 python /workspace/intelai_models/inference/fp32/eval.py --inter_op 1 --intra_op 28 --omp 28 --pipeline_config_path /checkpoints/rfcn_pipeline.config --checkpoint_dir /checkpoints --eval_dir /workspace/models/research/object_detection/models/rfcn/eval  --logtostderr  --blocktime=0  --run_once=True 
PYTHONPATH: :/workspace/intelai_models:/workspace/models/research:/workspace/models/research/slim:/workspace/models
RUNCMD: python common/tensorflow/run_tf_benchmark.py --framework=tensorflow --use-case=object_detection --model-name=rfcn --precision=fp32 --mode=inference --model-source-dir=/workspace/models --intelai-models=/workspace/intelai_models --num-cores=-1 --batch-size=1 --data-location=/dataset --socket-id 0 --verbose --checkpoint=/checkpoints         --config_file=rfcn_pipeline.config
Batch Size: 1
Ran inference with batch size 1
Log location outside container: {--output-dir value}/benchmark_rfcn_inference.log
```

And here is a sample log file tail when running for accuracy:
```
DONE (t=1.19s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.347
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.532
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.389
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.347
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.282
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.396
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.400
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.400
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
 Ran inference with batch size 1
Log location outside container: {--output-dir value}/benchmark_rfcn_inference_fp32_20181221_211905.log
```
