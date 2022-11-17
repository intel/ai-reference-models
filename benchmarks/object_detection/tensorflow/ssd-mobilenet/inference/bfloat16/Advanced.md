# SSD-MobileNet BFloat16 inference - Advanced instructions

SSD-MobileNet BFloat16 inference depends on Auto-Mixed-Precision to convert graph from FP32 to BFloat16 online.
Before evaluating SSD-MobileNet BFloat16 inference, please set the following environment variables:

```
export TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_ALLOWLIST_ADD=BiasAdd,Relu6,Mul,AddV2
export TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_INFERLIST_REMOVE=BiasAdd,AddV2,Mul
export TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_CLEARLIST_REMOVE=Relu6
```

1. Clone the `tensorflow/models` repository as `tensorflow-models` with the specified SHA,
since we are using an older version of the models repo for
SSD-MobileNet.

```
git clone https://github.com/tensorflow/models.git tensorflow-models
cd tensorflow-models
git checkout 20da786b078c85af57a4c88904f7889139739ab0
git clone https://github.com/cocodataset/cocoapi.git
```

The TensorFlow models repo will be used for running inference as well as
converting the coco dataset to the TF records format.

2. Follow the TensorFlow models object detection
[installation instructions](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/installation.md#installation)
to get your environment setup with the required dependencies.

3. Download and preprocess the COCO validation images using the [instructions here](/datasets/coco/README.md).
   Be sure to export the $OUTPUT_DIR environment variable.

4. Download the pretrained model:

```
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/ssdmobilenet_fp32_pretrained_model_combinedNMS.pb
```

5. Clone the [intelai/models](https://github.com/intelai/models) repo.
This repo has the launch script for running the model, which we will
use in the next step.

```
git clone https://github.com/IntelAI/models.git
```

6. Next, navigate to the `benchmarks` directory of the
[intelai/models](https://github.com/intelai/models) repo that was just
cloned in the previous step. SSD-MobileNet can be run for testing
online inference or testing accuracy.

To run for online inference, use the following command,
but replace in your path to the processed coco dataset images from step 3
for the `--dataset-location`, the path to the frozen graph that you
downloaded in step 4 as the `--in-graph`, and use the `--benchmark-only`
flag:

```
cd /home/<user>/models/benchmarks

python launch_benchmark.py \
    --data-location ${OUTPUT_DIR}/coco_val.record \
    --in-graph /home/<user>/ssdmobilenet_fp32_pretrained_model_combinedNMS.pb \
    --model-name ssd-mobilenet \
    --framework tensorflow \
    --precision bfloat16 \
    --mode inference \
    --socket-id 0 \
    --docker-image intel/intel-optimized-tensorflow:latest \
    --benchmark-only
```

To test accuracy, use the following command but replace in your path to
the tf record file that you generated for the `--data-location`,
the path to the frozen graph that you downloaded in step 4 as the
`--in-graph`, and use the `--accuracy-only` flag:

```
python launch_benchmark.py \
    --data-location ${OUTPUT_DIR}/coco_val.record \
    --in-graph /home/<user>/ssdmobilenet_fp32_pretrained_model_combinedNMS.pb \
    --model-name ssd-mobilenet \
    --framework tensorflow \
    --precision bfloat16 \
    --mode inference \
    --socket-id 0 \
    --docker-image intel/intel-optimized-tensorflow:latest \
    --accuracy-only
```
