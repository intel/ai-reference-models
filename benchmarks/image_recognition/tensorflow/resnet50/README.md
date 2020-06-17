# ResNet50

This document has instructions for how to run ResNet50 for the
following precisions:
* [Int8 inference](#int8-inference-instructions)
* [FP32 inference](#fp32-inference-instructions)

## Int8 Inference Instructions

These instructions use the TCMalloc memory allocator, which produces 
better performance results for Int8 precision models with smaller batch sizes. 
If you want to disable the use of TCMalloc, set `--disable-tcmalloc=True` 
when calling `launch_benchmark.py` and the script will run without TCMalloc.

1. Download the full ImageNet dataset and convert to the TF records format.

* Clone the tensorflow/models repository as tensorflow-models. This is to avoid conflict with Intel's `models` repo:
```
$ git clone https://github.com/tensorflow/models.git tensorflow-models
``` 
The TensorFlow models repo provides
[scripts and instructions](https://github.com/tensorflow/models/tree/master/research/slim#an-automated-script-for-processing-imagenet-data)
to download, process and convert the ImageNet dataset to the TF records format.

* The ImageNet dataset directory location is only required to calculate the model accuracy.

2. Download the pre-trained model.
```
$ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/resnet50_int8_pretrained_model.pb
```

3. Clone the 
[intelai/models](https://github.com/intelai/models)
repository
```
$ git clone https://github.com/IntelAI/models.git
```

4. Run the inference script `launch_benchmark.py` with the appropriate parameters to evaluate the model performance and/or calculate the accuracy.
The optimized ResNet50 model files are attached to the [intelai/models](https://github.com/intelai/models) repo and
located at `models/models/image_recognition/tensorflow/resnet50/`.

* Calculate the model accuracy, the required parameters parameters include: the `ImageNet` dataset location (from step 1),
the pre-trained `final_int8_resnet50.pb` input graph file (from step
2), and the `--accuracy-only` flag.
```
$ cd /home/<user>/models/benchmarks

$ python launch_benchmark.py \
    --data-location /home/<user>/dataset/FullImageNetData_directory
    --in-graph /home/<user>/resnet50_int8_pretrained_model.pb \
    --model-name resnet50 \
    --framework tensorflow \
    --precision int8 \
    --mode inference \
    --batch-size=100 \
    --accuracy-only \
    --docker-image intel/intel-optimized-tensorflow:2.1.0
```
The log file is saved to the value of `--output-dir`.

The tail of the log output when the script completes should look
something like this:
```
Iteration time: 233.495 ms
Processed 49600 images. (Top1 accuracy, Top5 accuracy) = (0.7361, 0.9155)
Iteration time: 233.231 ms
Processed 49700 images. (Top1 accuracy, Top5 accuracy) = (0.7361, 0.9155)
Iteration time: 234.541 ms
Processed 49800 images. (Top1 accuracy, Top5 accuracy) = (0.7360, 0.9154)
Iteration time: 233.033 ms
Processed 49900 images. (Top1 accuracy, Top5 accuracy) = (0.7361, 0.9155)
Iteration time: 233.013 ms
Processed 50000 images. (Top1 accuracy, Top5 accuracy) = (0.7360, 0.9154)
Ran inference with batch size 100
Log location outside container: {--output-dir value}/benchmark_resnet50_inference_int8_20190104_212224.log
```

* Evaluate the model performance: If just evaluate performance for dummy data, the `--data-location` is not needed.
Otherwise `--data-location` argument needs to be specified:
Calculate the model batch inference `images/sec`, the required parameters to run the inference script would include:
the pre-trained `resnet50_int8_pretrained_model.pb` input graph file (from step
2), and the `--benchmark-only` flag. It is
optional to specify the number of `warmup_steps` and `steps` as extra
args, as shown in the command below. If these values are not specified,
the script will default to use `warmup_steps=10` and `steps=50`.

```
$ cd /home/<user>/models/benchmarks

$ python launch_benchmark.py \
    --in-graph /home/<user>/resnet50_int8_pretrained_model.pb \
    --model-name resnet50 \
    --framework tensorflow \
    --precision int8 \
    --mode inference \
    --batch-size=128 \
    --benchmark-only \
    --docker-image intel/intel-optimized-tensorflow:2.1.0
    -- warmup_steps=50 steps=500
```
The tail of the log output when the script completes should look
something like this:
```
...
Iteration 497: 0.253495 sec
Iteration 498: 0.253033 sec
Iteration 499: 0.258083 sec
Iteration 500: 0.254541 sec
Average time: 0.254572 sec
Batch size = 128
Throughput: 502.805 images/sec
Ran inference with batch size 128
Log location outside container: {--output-dir value}/benchmark_resnet50_inference_int8_20190416_172735.log
```

Note that the `--verbose` or `--output-dir` flag can be added to any of the above commands
to get additional debug output or change the default output location..

## FP32 Inference Instructions

1. Download the pre-trained model.
```
$ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/resnet50_fp32_pretrained_model.pb
```

2. Clone the 
[intelai/models](https://github.com/intelai/models)
repository
```
$ git clone https://github.com/IntelAI/models.git
```

3. If running resnet50 for accuracy, the ImageNet dataset will be
required (if running for batch or online inference performance, then dummy
data will be used).

The TensorFlow models repo provides
[scripts and instructions](https://github.com/tensorflow/models/tree/master/research/slim#an-automated-script-for-processing-imagenet-data)
to download, process, and convert the ImageNet dataset to the TF records format.

4. Run the inference script `launch_benchmark.py` with the appropriate parameters to evaluate the model performance.
The optimized ResNet50 model files are attached to the [intelai/models](https://github.com/intelai/models) repo and
located at `models/models/image_recognition/tensorflow/resnet50/`.
If using dummy data for inference, `--data-location` flag is not required. Otherwise,
`--data-location` needs to point to point to ImageNet dataset location.

* To measure online inference, set `--batch-size=1` and run the script as shown:
```
$ cd /home/<user>/models/benchmarks

$ python launch_benchmark.py \
    --in-graph /home/<user>/resnet50_fp32_pretrained_model.pb \
    --model-name resnet50 \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --batch-size=1 \
    --socket-id 0 \
    --docker-image intel/intel-optimized-tensorflow:2.1.0
```

The log file is saved to the value of `--output-dir`.

The tail of the log output when the script completes should look
something like this:
```
Inference with dummy data.
Iteration 1: 0.956 sec
Iteration 2: 0.018 sec
...
Iteration 38: 0.011 sec
Iteration 39: 0.011 sec
Iteration 40: 0.011 sec
Average time: 0.011 sec
Batch size = 1
Latency: 10.924 ms
Throughput: 91.541 images/sec
Ran inference with batch size 1
Log location outside container: {--output-dir value}/benchmark_resnet50_inference_fp32_20190104_215326.log
```

* To measure batch inference, set `--batch-size=128` and run the launch script as shown:
```
$ cd /home/<user>/models/benchmarks

$ python launch_benchmark.py \
    --in-graph /home/<user>/resnet50_fp32_pretrained_model.pb \
    --model-name resnet50 \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --batch-size=128 \
    --socket-id 0 \
    --docker-image intel/intel-optimized-tensorflow:2.1.0
```

The log file is saved to the value of `--output-dir`.

The tail of the log output when the script completes should look
something like this:
```
Inference with dummy data.
Iteration 1: 1.777 sec
Iteration 2: 0.657 sec
Iteration 3: 0.652 sec
...
Iteration 38: 0.653 sec
Iteration 39: 0.649 sec
Iteration 40: 0.652 sec
Average time: 0.653 sec
Batch size = 128
Throughput: 196.065 images/sec
Ran inference with batch size 128
Log location outside container: {--output-dir value}/benchmark_resnet50_inference_fp32_20190104_215655.log
```

* To measure the model accuracy, use the `--accuracy-only` flag and pass
the ImageNet dataset directory from step 3 as the `--data-location`:
```
$ cd /home/<user>/models/benchmarks

$ python launch_benchmark.py \
    --in-graph /home/<user>/resnet50_fp32_pretrained_model.pb \
    --model-name resnet50 \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --accuracy-only \
    --batch-size 100 \
    --socket-id 0 \
    --data-location /home/<user>/dataset/ImageNetData_directory \
    --docker-image intel/intel-optimized-tensorflow:2.1.0
```

The log file is saved to the value of `--output-dir`.
The tail of the log output when the accuracy run completes should look
something like this:
```
...
Iteration time: 649.252 ms
Processed 50000 images. (Top1 accuracy, Top5 accuracy) = (0.7430, 0.9188)
Ran inference with batch size 100
Log location outside container: {--output-dir value}/benchmark_resnet50_inference_fp32_20190104_213452.log
```

* The `--output-results` flag can be used to also output a file with the inference
results (file name, actual label, and the predicted label). The results
output can only be used with real data.

For example, the command below is the same as the accuracy test above,
except with the `--output-results` flag added:
```
$ cd /home/<user>/models/benchmarks

$ python launch_benchmark.py \
    --in-graph /home/<user>/resnet50_fp32_pretrained_model/freezed_resnet50.pb \
    --model-name resnet50 \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --accuracy-only \
    --output-results \
    --batch-size 100 \
    --socket-id 0 \
    --data-location /home/<user>/dataset/ImageNetData_directory \
    --docker-image intel/intel-optimized-tensorflow:2.1.0
```
The results file will be written to the
`models/benchmarks/common/tensorflow/logs` directory, unless another
output directory is specified by the `--output-dir` arg. Below is an
example of what the inference results file will look like:
```
filename,actual,prediction
ILSVRC2012_val_00025616.JPEG,96,96
ILSVRC2012_val_00037570.JPEG,656,656
ILSVRC2012_val_00038006.JPEG,746,746
ILSVRC2012_val_00023384.JPEG,413,793
ILSVRC2012_val_00014392.JPEG,419,419
ILSVRC2012_val_00015258.JPEG,740,740
ILSVRC2012_val_00042399.JPEG,995,995
ILSVRC2012_val_00022226.JPEG,528,528
ILSVRC2012_val_00021512.JPEG,424,424
...
```

Note that the `--verbose` or `--output-dir` flag can be added to any of the above commands
to get additional debug output or change the default output location.
