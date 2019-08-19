# RFCN and SSD-MobileNet

This document has instructions for how to run RFCN and SSD-MobileNet for the
following modes/precisions:
* [FP32 inference](#fp32-inference-instructions)

## FP32 Inference Instructions

1. Store the path to the current directory and clone this [intelai/models](https://github.com/IntelAI/models)
repository:

```
$ MODEL_WORK_DIR=${MODEL_WORK_DIR:=`pwd`}
$ pushd $MODEL_WORK_DIR

$ git clone https://github.com/IntelAI/models.git
```
2. Download the 2017 validation COCO dataset:
>Note: do not convert the COCO dataset to TF records format
```
$ mkdir -p coco/val
$ wget http://images.cocodataset.org/zips/val2017.zip
$ unzip val2017.zip -d coco/val
```

3. Choose your model and download the pre-trained SavedModel: Select one of the models (R-FCN or SSD-MobileNet). Then download and extract the pre-trained model and copy the saved_model.pb
Highlight and copy one of the following download links:

* R-FCN:
```
$ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/rfcn_resnet101_fp32_coco_pretrained_model.tar.gz
$ tar -xzvf rfcn_resnet101_fp32_coco_pretrained_model.tar.gz
```
* SSD-MobileNet:
```
$ wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz
$ tar -xzvf ssd_mobilenet_v1_coco_2018_01_28.tar.gz
```

The saved graphs `$MODEL_WORK_DIR/rfcn_resnet101_coco_2018_01_28/saved_model/saved_model.pb` and `$MODEL_WORK_DIR/ssd_mobilenet_v1_coco_2018_01_28/saved_model/saved_model.pb` will be used as `--in-graph` later to run the chosen model test in step 6.

4. Clone the TensorFlow models repository into a new folder in your home directory:
```
$ git clone https://github.com/tensorflow/models tensorflow-models
$ export TF_MODELS_ROOT=$(pwd)/tensorflow-models
```

5. Navigate to the `benchmarks` directory in your local clone of
the [intelai/models](https://github.com/IntelAI/models) repo from step 1.
The `launch_benchmark.py` script in the `benchmarks` directory is
used for starting a tensorflow serving run using optimized TensorFlow Serving docker
container. It has arguments to specify which model, framework, mode,
precision, and input graph.
```
$ cd models/benchmarks
```

6. Both models can be run for measuring batch or online inference performance. Use the following examples below,
depending on your model.

Test online inference and batch inference with COCO dataset (using one of the models `rfcn` or `ssd-mobilenet` as `--model-name` , `--data-location` from step 2, `--in-graph` from step 3, and `--model-source-dir` from step 4):

>Note: `GRPC` protocol is used by default in order to try `REST`, please set `export PROTOCOL_NAME="rest"`

* For `RFCN`:
```
$ python launch_benchmark.py \
    --in-graph $MODEL_WORK_DIR/rfcn_resnet101_coco_2018_01_28/saved_model/saved_model.pb \
    --model-name rfcn \
    --framework tensorflow_serving \
    --model-source-dir $TF_MODELS_ROOT \
    --data-location $MODEL_WORK_DIR/coco/val/val2017 \
    --precision fp32 \
    --mode inference \
    --benchmark-only
```

Example log tail when running for online inference and batch inference with COCO dataset for `RFCN`:
```
 SERVER_URL: localhost:8500 
 IMAGES_PATH: ~/coco/val/val2017

Starting RFCN model benchmarking for latency on GRPC:
batch_size=1, num_iteration=20, warm_up_iteration=10

Iteration 1: .. sec
...
Average time: .. sec
Batch size = 1
Latency: .. ms
Throughput: .. images/sec

Starting RFCN model benchmarking for throughput on GRPC:
batch_size=128, num_iteration=10, warm_up_iteration=2

Iteration 1: .. sec
...
Average time: .. sec
Batch size = 128
Throughput: .. images/sec
Log output location: ~/models/benchmarks/common/tensorflow_serving/logs/benchmark_rfcn_inference_fp32_20190815_112142.log
```

* For `SSD-MobileNet`:
```
$ python launch_benchmark.py \
    --in-graph $MODEL_WORK_DIR/ssd_mobilenet_v1_coco_2018_01_28/saved_model/saved_model.pb \
    --model-name ssd-mobilenet \
    --framework tensorflow_serving \
    --model-source-dir $TF_MODELS_ROOT \
    --data-location $MODEL_WORK_DIR/coco/val/val2017 \
    --precision fp32 \
    --mode inference \
    --benchmark-only
```
Example log tail when running for online inference and batch inference with COCO dataset for `SSD-MobileNet`:
```
 SERVER_URL: localhost:8500 
 IMAGES_PATH: ~/coco/val/val2017

Starting SSDMOBILENET model benchmarking for latency on GRPC:
batch_size=1, num_iteration=20, warm_up_iteration=10

Iteration 1: .. sec
...
Iteration 20: .. sec
Average time: .. sec
Batch size = 1
Latency: .. ms
Throughput: .. images/sec

Starting SSDMOBILENET model benchmarking for throughput on GRPC:
batch_size=128, num_iteration=10, warm_up_iteration=2

Iteration 1: .. sec
...
Iteration 10: .. sec
Average time: .. sec
Batch size = 128
Throughput: .. images/sec
Log output location: ~/models/benchmarks/common/tensorflow_serving/logs/benchmark_ssd-mobilenet_inference_fp32_20190815_120325.log
```


>Note that the `--verbose` or `--output-dir` flag can be added to any of the above commands
to get additional debug output or change the default output location.

7. To return to where you started from:
```
$ popd
```

For more details about using object detection models with TensorFlow serving, please check [Object Detection with TensorFlow Serving on CPU](/docs/object_detection/tensorflow_serving/Tutorial.md)