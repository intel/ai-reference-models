# SSD-MobileNet

This document has instructions for how to run SSD-MobileNet for the
following modes/platforms:
* [FP32 inference](#fp32-inference-instructions)

Benchmarking instructions and scripts for model training and inference
other platforms are coming later.

## FP32 Inference Instructions

1. Download and convert the dataset

This example uses the [COCO dataset](http://cocodataset.org/#home).
**_TODO: Find out exactly which file Lakshay used (there are different
datasets for different years, etc)._**
Once you have downloaded the COCO dataset using the link provided, use
the `create_coco_tf_record.py` file to convert the raw COCO dataset to
the TFRecord format in order to use it with object detection:
https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_coco_tf_record.py

Once the dataset has been converted, you should have a file something
like: `coco_val.record`.

2. Download the pre-trained model

**_TODO: Find out from Lakshay if he downloaded the pre-trained model from
here and if so, exactly which one was used._**

Download a pre-trained SSD-MobileNet model from the
[TensorFlow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#coco-trained-models).
The downloaded .tar file includes a `frozen_inference_graph.pb` which we
will be using when running inference.

3. Clone the `tensorflow-models` repository, if you don't already have
it:

```
$ git clone git@github.com:tensorflow/models.git
```

4. In another directory, clone this `intel-models` repository and then
navigate to the `benchmark` folder:

```
$ git clone git@github.com:NervanaSystems/intel-models.git

$ cd benchmarks
```

5. Run the `launch_benchmark.py` script with the appropriate parameters
including: the `coco_val.record` data location (from step 1),
the pre-trained `frozen_inference_graph.pb` input graph file (from step
2), and the location of your `tensorflow/models` clone (from step 3).

```
$ python launch_benchmark.py \
    --data-location /home/aipguser/dina/SSD-Mobilenet-graph/data/coco_val.record \
    --in-graph /home/aipguser/dina/SSD-Mobilenet-graph/graph/frozen_inference_graph.pb \
    --model-source-dir /home/aipguser/dina/models \
    --model-name ssd-mobilenet \
    --framework tensorflow \
    --platform fp32 \
    --mode inference \
    --docker-image intelaipg/intel-optimized-tensorflow:latest
```

6. The output of job should look something like this snippet:

```
INFO:tensorflow:Processed 4970 images...
INFO:tensorflow:Processed 4980 images...
INFO:tensorflow:Processed 4990 images...
INFO:tensorflow:Processed 5000 images...
INFO:tensorflow:Finished processing records
Using model init: /workspace/benchmark/tensorflow/ssd-mobilenet/fp32/inference/model_init.py
Received these standard args: Namespace(batch_size=256, checkpoint=None, data_location='/dataset', inference_only=True, input_graph='/in_graph/frozen_inference_graph.pb', mode='inference', model_name='ssd-mobilenet', model_source_dir='/workspace/models', num_cores=-1, num_inter_threads=2, num_intra_threads=56, platform='fp32', single_socket=True, socket_id=0, verbose=True)
Received these custom args: []
Initialize here.
Run model here.
current directory: /workspace/models/research
Running: OMP_NUM_THREADS=28 numactl -l -N 1 python object_detection/inference/infer_detections.py --input_tfrecord_paths /dataset --inference_graph /in_graph/frozen_inference_graph.pb --output_tfrecord_path=/tmp/ssd-mobilenet-record-out --intra_op_parallelism_threads 28 --inter_op_parallelism_threads 1 --discard_image_pixels=True --inference_only
```
