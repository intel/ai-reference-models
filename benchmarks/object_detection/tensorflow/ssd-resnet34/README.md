# SSD-ResNet34

The following links have instructions for how to run SSD-ResNet34 for the
following modes/precisions:
* [FP32 inference](/benchmarks/object_detection/tensorflow/ssd-resnet34/inference/fp32/README.md)
* [BF16 inference](#bf16-inference-instructions)
* [INT8 inference](/benchmarks/object_detection/tensorflow/ssd-resnet34/inference/int8/README.md)
* [FP32 Training](/benchmarks/object_detection/tensorflow/ssd-resnet34/training/fp32/README.md)
* [BF16 Training](/benchmarks/object_detection/tensorflow/ssd-resnet34/training/bfloat16/README.md)


## BF16 Inference Instructions
1. Please ensure you have installed all the libraries listed in the
`requirements` before you start the next step.

2. Clone the `tensorflow/models` repository with the specified SHA,
since we are using an older version of the models repo for
SSD-ResNet34.

```
git clone https://github.com/tensorflow/models.git tf_models
cd tf_models
git checkout f505cecde2d8ebf6fe15f40fb8bc350b2b1ed5dc
git clone https://github.com/cocodataset/cocoapi.git
```

The TensorFlow models repo will be used for running inference as well as
converting the coco dataset to the TF records format.

3. Follow the TensorFlow models object detection
[installation instructions](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/installation.md#installation)
to get your environment setup with the required dependencies.

4. Download and preprocess the COCO validation images using the [instructions here](/datasets/coco/README.md).
   Be sure to export the $DATASET_DIR and $OUTPUT_DIR environment variables. Then, rename the tf_records file and copy the annotations file:

```
mv ${OUTPUT_DIR}/coco_val.record ${OUTPUT_DIR}/validation-00000-of-00001
cp -r ${DATASET_DIR}/annotations ${OUTPUT_DIR}
```

5. Download the pretrained model:

```
# ssd-resnet34 300x300
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/ssd_resnet34_fp32_bs1_pretrained_model.pb

# ssd-resnet34 1200x1200
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/ssd_resnet34_fp32_1200x1200_pretrained_model.pb
```

6. Clone the [intelai/models](https://github.com/intelai/models) repo.
This repo has the launch script for running the model, which we will
use in the next step.

```
git clone https://github.com/IntelAI/models.git
```

7. Clone the [tensorflow/benchmarks](https://github.com/tensorflow/benchmarks.git) repo. This repo contains the method needed
to run the ssd-resnet34 model. Please ensure that the `ssd-resnet-benchmarks` and `models` repos are in the same folder.

```
git clone --single-branch https://github.com/tensorflow/benchmarks.git ssd-resnet-benchmarks
cd ssd-resnet-benchmarks
git checkout 509b9d288937216ca7069f31cfb22aaa7db6a4a7
cd ../
```

8. Next, navigate to the `benchmarks` directory of the
[intelai/models](https://github.com/intelai/models) repo that was cloned. Use the following commands to measure performance and accuracy.

To run with input size of 1200x1200, set the `--in-graph` to the downloaded frozen graph `ssd_resnet34_fp32_1200x1200_pretrained_model.pb` and add `-- input-size=1200` flag (by default, the benchmark runs with input size of 300x300).
To run with input size of 300x300, set the `--in-graph` to the downloaded frozen graph `ssd_resnet34_fp32_bs1_pretrained_model.pb`.
Use `--benchmark-only` flag to measure performance, and `--accuracy-only` flag to test accuracy.
If you run in Docker mode, you also need to provide `ssd-resnet-benchmarks` path for `volume` flag.

```
cd $MODEL_WORK_DIR/models/benchmarks

# benchmarks with input size 1200x1200
python launch_benchmark.py \
    --in-graph /home/<user>/ssd_resnet34_fp32_1200x1200_pretrained_model.pb \
    --model-source-dir /home/<user>/tf_models \
    --model-name ssd-resnet34 \
    --framework tensorflow \
    --precision bfloat16 \
    --mode inference \
    --socket-id 0 \
    --batch-size 1 \
    --docker-image intel/intel-optimized-tensorflow:latest \
    --volume /home/<user>/ssd-resnet-benchmarks:/workspace/ssd-resnet-benchmarks \
    --benchmark-only \
    -- input-size=1200
```

To run the accuracy test, use the following command with `--data-location` set to the tf record file that you generated.

```
# accuracy test with input size 1200x1200
python launch_benchmark.py \
    --data-location <path_to_tf_records> \
    --in-graph /home/<user>/ssd_resnet34_fp32_1200x1200_pretrained_model.pb \
    --model-source-dir /home/<user>/tf_models \
    --model-name ssd-resnet34 \
    --framework tensorflow \
    --precision bfloat16 \
    --mode inference \
    --socket-id 0 \
    --batch-size 1 \
    --docker-image intel/intel-optimized-tensorflow:latest \
    --volume /home/<user>/ssd-resnet-benchmarks:/workspace/ssd-resnet-benchmarks \
    --accuracy-only \
    -- input-size=1200
```

9. The log file is saved to the value of `--output-dir`.

Below is a sample log file tail when running for performance:

```
Batchsize: 1
Time spent per BATCH:    ... ms
Total samples/sec:    ... samples/s
```

Below is a sample log file tail when testing accuracy:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.224
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.410
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.220
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.140
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.297
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.257
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.214
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.342
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.367
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.185
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.454
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.443
```
