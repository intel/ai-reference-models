# Faster R-CNN (ResNet50)

This document has instructions for how to run Faster R-CNN for the
following modes/precisions:
* [FP32 inference](#fp32-inference-instructions)
* [Int8 inference](#int8-inference-instructions)

Instructions and scripts for the Faster R-CNN ResNet50 model training and inference
for other precisions are coming later.

## FP32 Inference Instructions

1. Clone the `tensorflow/models` at the specified tag. Scripts from this
   repo will be used for running inference.
   ```
   git clone https://github.com/tensorflow/models.git
   pushd models
   git checkout tags/v1.12.0
   popd
   ```

2. Download and preprocess the COCO validation images using the [instructions here](/datasets/coco/README.md).
   Be sure to export the $DATASET_DIR and $OUTPUT_DIR environment variables.
   The `coco_val.record` file will be used in the following inference examples.

3. Download and extract the pre-trained model.
   ```
   wget https://storage.googleapis.com/intel-optimized-tensorflow/models/faster_rcnn_resnet50_fp32_coco_pretrained_model.tar.gz
   tar -xzvf faster_rcnn_resnet50_fp32_coco_pretrained_model.tar.gz
   ```

4. Clone the [intelai/models](https://github.com/intelai/models) repo.
   This repo has the launch script for running the model.
   ```
   git clone https://github.com/IntelAI/models.git
   ```

5. Run the `launch_benchmark.py` script from the intelai/models repo,
   with the appropriate parameters including: the `coco_val.record` data
   location (from step 2), the pre-trained model `pipeline.config` file
   and the checkpoint location (from step 3), and the location of your
   `tensorflow/models` clone (from step 1).

   Run batch and online inference using the following command:
   ```
   cd /home/<user>/models/benchmarks

   python launch_benchmark.py \
     --data-location $OUTPUT_DIR \
     --model-source-dir /home/<user>/tensorflow/models \
     --model-name faster_rcnn \
     --framework tensorflow \
     --precision fp32 \
     --mode inference \
     --socket-id 0 \
     --checkpoint /home/<user>/faster_rcnn_resnet50_fp32_coco \
     --docker-image intelaipg/intel-optimized-tensorflow:1.15.2 \
     -- config_file=pipeline.config
    ```

   Or test accuracy where the `--data-location` is the path the directory
   where your `coco_val.record` file is located and the `--in-graph` is
   the pre-trained graph located in the pre-trained model directory (from step 3):
   ```
   python launch_benchmark.py \
     --model-name faster_rcnn \
     --mode inference \
     --precision fp32 \
     --framework tensorflow \
     --docker-image intelaipg/intel-optimized-tensorflow:1.15.2 \
     --model-source-dir /home/<user>/tensorflow/models \
     --data-location $OUTPUT_DIR \
     --in-graph /home/<user>/faster_rcnn_resnet50_fp32_coco/frozen_inference_graph.pb \
     --accuracy-only
   ```

6. The log file is saved to the value of `--output-dir`. Below is a
   sample log file tail when running for batch and online inference:

   ```
   I0923 18:13:53.033420 140562203281216 eval_util.py:72] Metrics written to tf summary.
   I0923 18:13:53.033456 140562203281216 eval_util.py:463] Finished evaluation!
   Time spent : ... seconds.
   Time spent per BATCH: ... seconds.
   Log file location: {--output-dir value}/benchmark_faster_rcnn_inference_fp32_20200923_181013.log
    ```

   And here is a sample log file tail when running for accuracy:
    ```
    DONE (t=1.35s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.316
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.489
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.355
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.316
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.271
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.380
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.383
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.383
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
    Ran inference with batch size 1
    Log location outside container: {--output-dir value}/benchmark_faster_rcnn_inference_fp32_20190114_205714.log
    ```

## Int8 Inference Instructions

These instructions use the TCMalloc memory allocator, which produces 
better performance results for Int8 precision models with smaller batch sizes. 
If you want to disable the use of TCMalloc, set `--disable-tcmalloc=True` 
when calling `launch_benchmark.py` and the script will run without TCMalloc.

1. Please follow the steps from the
   [Faster R-CNN FP32 instructions](#fp32-inference-instructions) written
   above for cloning dependecy repositories and getting the coco dataset
   (steps 1 and 2). Note that the performance test uses the raw coco
   images, but the accuracy test uses the TF records file.

2. Download the pre-trained model.
   ```
   wget https://storage.googleapis.com/intel-optimized-tensorflow/models/faster_rcnn_int8_pretrained_model.pb
   ```

3. Clone the [intelai/models](https://github.com/intelai/models) repo.
   This repo has the launch script for running the model.
   ```
   git clone https://github.com/IntelAI/models.git
   ```

4. Run the `launch_benchmark.py` script from the intelai/models repo,
   with the appropriate parameters. To run on single socket use
   `--socket_id` switch. By default, it will be using all available
   sockets. The optional parameter `number_of_steps` (default value =
   5000) can be added at the end of command after `--` as shown below:

   Run batch and online inference using the following command. The
   `--data-location` is the path to the directory that contains the raw
   coco dataset validation images which you downloaded and unzipped:
   ```
   cd /home/<user>/models/benchmarks

   python launch_benchmark.py \
     --data-location ${DATASET_DIR} \
     --model-source-dir /home/<user>/tensorflow/models \
     --model-name faster_rcnn \
     --framework tensorflow \
     --precision int8 \
     --mode inference \
     --socket-id 0 \
     --in-graph /home/<user>/faster_rcnn_int8_pretrained_model.pb \
     --docker-image intelaipg/intel-optimized-tensorflow:1.15.2 \
     --benchmark-only \
     -- number_of_steps=5000
    ```

   Test accuracy where the `--data-location` is the path the directory
   where your `coco_val.record` file is located and the `--in-graph` is
   the pre-trained graph model:
   ```
   python launch_benchmark.py \
     --model-name faster_rcnn \
     --mode inference \
     --precision int8 \
     --framework tensorflow \
     --socket-id 0 \
     --docker-image intelaipg/intel-optimized-tensorflow:1.15.2 \
     --model-source-dir /home/<user>/tensorflow/models \
     --data-location ${OUTPUT_DIR}/coco_val.record \
     --in-graph /home/<user>/faster_rcnn_int8_pretrained_model.pb  \
     --accuracy-only
   ```

5. The log file is saved to the value of `--output-dir`. Below is a
   sample log file tail when running for batch and online inference:
   ```
   Step 4970: ... seconds
   Step 4980: ... seconds
   Step 4990: ... seconds
   Avg. Duration per Step: ...
   Log location outside container: {--output-dir value}/benchmark_faster_rcnn_inference_int8_20190117_232539.log
   ```

   And here is a sample log file tail when running for accuracy:
   ```
   Accumulating evaluation results...
   DONE (t=1.34s).
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.310
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.479
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.351
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.310
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.267
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.372
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.375
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.375
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
   Ran inference with batch size -1
   Log location outside container: {--output-dir value}/benchmark_faster_rcnn_inference_int8_20190117_231937.log
   ```
