<!-- 50. Launch benchmark instructions -->
Once your environment is setup, navigate to the `benchmarks` directory of
the model zoo and set environment variables pointing to the directory for the
dataset, pretrained model frozen graph, and an output directory where
log files will be written.

```
# cd to the benchmarks directory in the model zoo
cd benchmarks

export PRETRAINED_MODEL=<path to the downloaded frozen graph>
export DATASET_DIR=<path to the coco tf record file>
export OUTPUT_DIR=<directory where log files will be written>
```

SSD-MobileNet can be run for testing online inference or testing accuracy.

* To run for online inference, use the following command:
  ```
  python launch_benchmark.py \
    --data-location ${DATASET_DIR} \
    --in-graph ${PRETRAINED_MODEL} \
    --model-name ssd-mobilenet \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --socket-id 0 \
    --docker-image <docker image> \
    --output-dir ${OUTPUT_DIR} \
    --benchmark-only
  ```
* To test accuracy, use the following command:
  ```
  python launch_benchmark.py \
    --data-location ${DATASET_DIR} \
    --in-graph ${PRETRAINED_MODEL} \
    --model-name ssd-mobilenet \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --socket-id 0 \
    --docker-image <docker image> \
    --output-dir ${OUTPUT_DIR} \
    --accuracy-only
  ```

The log file is saved to the value of `${OUTPUT_DIR}`.

Below is a sample log file tail when running for performance:
```
INFO:tensorflow:Processed 5001 images... moving average latency 37 ms
INFO:tensorflow:Finished processing records
Latency: min = 33.8, max = 6635.9, mean= 38.4, median = 37.2
Ran inference with batch size -1
Log location outside container: ${OUTPUT_DIR}/benchmark_ssd-mobilenet_inference_fp32_20190130_225108.log
```

Below is a sample log file tail when testing accuracy:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.231
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.349
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.254
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.231
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.209
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.264
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.264
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.264
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
Ran inference with batch size -1
Log location outside container: ${OUTPUT_DIR}/benchmark_ssd-mobilenet_inference_fp32_20190123_225145.log
```

Batch and online inference can also be run with multiple instances using
`numactl`. The following commands have examples how to do multi-instance runs
using the `--numa-cores-per-instance` argument. Note that these examples are
running with synthetic data (to use real data, you can add `--data-location ${DATASET_DIR}`).
Your output may vary from what's seen below, depending on the number of
cores on your system.

* For multi-instance batch inference, the recommended configuration uses all
  the cores on a socket for each instance (this means that if you have 2 sockets,
  you would be running 2 instances - one per socket) and a batch size of 448.
  ```
  python launch_benchmark.py \
    --in-graph ${PRETRAINED_MODEL} \
    --output-dir ${OUTPUT_DIR} \
    --model-name ssd-mobilenet \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --batch-size 448 \
    --numa-cores-per-instance socket \
    --docker-image <docker image> \
    --benchmark-only
  ```
  The output will show the multi-instance command being run and a list of the
  log files (one for each instance and a combined log file):
  ```
  Multi-instance run:
  OMP_NUM_THREADS=28 numactl --localalloc --physcpubind=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27 python /workspace/intelai_models/inference/infer_detections.py -g /in_graph/ssdmobilenet_fp32_pretrained_model_combinedNMS.pb -i 1000 -w 200 -a 28 -e 1 -b 448 >> ${OUTPUT_DIR}/ssd-mobilenet_fp32_inference_bs1_cores28_instance0.log 2>&1 & \
  OMP_NUM_THREADS=28 numactl --localalloc --physcpubind=28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55 python /workspace/intelai_models/inference/infer_detections.py -g /in_graph/ssdmobilenet_fp32_pretrained_model_combinedNMS.pb -i 1000 -w 200 -a 28 -e 1 -b 448 >> ${OUTPUT_DIR}/ssd-mobilenet_fp32_inference_bs1_cores28_instance1.log 2>&1 & \
  wait

  The following log files were saved to the output directory:
  ssd-mobilenet_fp32_inference_bs448_cores28_instance0.log
  ssd-mobilenet_fp32_inference_bs448_cores28_instance1.log

  A combined log file was saved to the output directory:
  ssd-mobilenet_fp32_inference_bs1_cores28_all_instances.log
  ```
  The following grep command can be used to get a summary of the total samples
  per second from the combined log file:
  ```
  grep 'Total samples/sec' ${OUTPUT_DIR}/ssd-mobilenet_fp32_inference_bs448_cores*_all_instances.log  | awk -F' ' '{sum+=$3;} END{print sum} '
  ```
* For multi-instance online inference, the recommended configuration is using
  4 cores per instance and a batch size of 1.
  ```
  python launch_benchmark.py \
    --in-graph ${PRETRAINED_MODEL} \
    --output-dir ${OUTPUT_DIR} \
    --model-name ssd-mobilenet \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --batch-size 1 \
    --numa-cores-per-instance 4 \
    --docker-image <docker image> \
    --benchmark-only
  ```
  The output will show the multi-instance command being run and a list of the
  log files (one for each instance and a combined log file):
  ```
  Multi-instance run:
  OMP_NUM_THREADS=4 numactl --localalloc --physcpubind=0,1,2,3 python /workspace/intelai_models/inference/infer_detections.py -g /in_graph/ssdmobilenet_fp32_pretrained_model_combinedNMS.pb -i 1000 -w 200 -a 4 -e 1 -b 1 >> ${OUTPUT_DIR}/ssd-mobilenet_fp32_inference_bs1_cores4_instance0.log 2>&1 & \
  OMP_NUM_THREADS=4 numactl --localalloc --physcpubind=4,5,6,7 python /workspace/intelai_models/inference/infer_detections.py -g /in_graph/ssdmobilenet_fp32_pretrained_model_combinedNMS.pb -i 1000 -w 200 -a 4 -e 1 -b 1 >> ${OUTPUT_DIR}/ssd-mobilenet_fp32_inference_bs1_cores4_instance1.log 2>&1 & \
  OMP_NUM_THREADS=4 numactl --localalloc --physcpubind=8,9,10,11 python /workspace/intelai_models/inference/infer_detections.py -g /in_graph/ssdmobilenet_fp32_pretrained_model_combinedNMS.pb -i 1000 -w 200 -a 4 -e 1 -b 1 >> ${OUTPUT_DIR}/ssd-mobilenet_fp32_inference_bs1_cores4_instance2.log 2>&1 & \
  OMP_NUM_THREADS=4 numactl --localalloc --physcpubind=12,13,14,15 python /workspace/intelai_models/inference/infer_detections.py -g /in_graph/ssdmobilenet_fp32_pretrained_model_combinedNMS.pb -i 1000 -w 200 -a 4 -e 1 -b 1 >> ${OUTPUT_DIR}/ssd-mobilenet_fp32_inference_bs1_cores4_instance3.log 2>&1 & \
  OMP_NUM_THREADS=4 numactl --localalloc --physcpubind=16,17,18,19 python /workspace/intelai_models/inference/infer_detections.py -g /in_graph/ssdmobilenet_fp32_pretrained_model_combinedNMS.pb -i 1000 -w 200 -a 4 -e 1 -b 1 >> ${OUTPUT_DIR}/ssd-mobilenet_fp32_inference_bs1_cores4_instance4.log 2>&1 & \
  OMP_NUM_THREADS=4 numactl --localalloc --physcpubind=20,21,22,23 python /workspace/intelai_models/inference/infer_detections.py -g /in_graph/ssdmobilenet_fp32_pretrained_model_combinedNMS.pb -i 1000 -w 200 -a 4 -e 1 -b 1 >> ${OUTPUT_DIR}/ssd-mobilenet_fp32_inference_bs1_cores4_instance5.log 2>&1 & \
  OMP_NUM_THREADS=4 numactl --localalloc --physcpubind=24,25,26,27 python /workspace/intelai_models/inference/infer_detections.py -g /in_graph/ssdmobilenet_fp32_pretrained_model_combinedNMS.pb -i 1000 -w 200 -a 4 -e 1 -b 1 >> ${OUTPUT_DIR}/ssd-mobilenet_fp32_inference_bs1_cores4_instance6.log 2>&1 & \
  OMP_NUM_THREADS=4 numactl --localalloc --physcpubind=28,29,30,31 python /workspace/intelai_models/inference/infer_detections.py -g /in_graph/ssdmobilenet_fp32_pretrained_model_combinedNMS.pb -i 1000 -w 200 -a 4 -e 1 -b 1 >> ${OUTPUT_DIR}/ssd-mobilenet_fp32_inference_bs1_cores4_instance7.log 2>&1 & \
  OMP_NUM_THREADS=4 numactl --localalloc --physcpubind=32,33,34,35 python /workspace/intelai_models/inference/infer_detections.py -g /in_graph/ssdmobilenet_fp32_pretrained_model_combinedNMS.pb -i 1000 -w 200 -a 4 -e 1 -b 1 >> ${OUTPUT_DIR}/ssd-mobilenet_fp32_inference_bs1_cores4_instance8.log 2>&1 & \
  OMP_NUM_THREADS=4 numactl --localalloc --physcpubind=36,37,38,39 python /workspace/intelai_models/inference/infer_detections.py -g /in_graph/ssdmobilenet_fp32_pretrained_model_combinedNMS.pb -i 1000 -w 200 -a 4 -e 1 -b 1 >> ${OUTPUT_DIR}/ssd-mobilenet_fp32_inference_bs1_cores4_instance9.log 2>&1 & \
  OMP_NUM_THREADS=4 numactl --localalloc --physcpubind=40,41,42,43 python /workspace/intelai_models/inference/infer_detections.py -g /in_graph/ssdmobilenet_fp32_pretrained_model_combinedNMS.pb -i 1000 -w 200 -a 4 -e 1 -b 1 >> ${OUTPUT_DIR}/ssd-mobilenet_fp32_inference_bs1_cores4_instance10.log 2>&1 & \
  OMP_NUM_THREADS=4 numactl --localalloc --physcpubind=44,45,46,47 python /workspace/intelai_models/inference/infer_detections.py -g /in_graph/ssdmobilenet_fp32_pretrained_model_combinedNMS.pb -i 1000 -w 200 -a 4 -e 1 -b 1 >> ${OUTPUT_DIR}/ssd-mobilenet_fp32_inference_bs1_cores4_instance11.log 2>&1 & \
  OMP_NUM_THREADS=4 numactl --localalloc --physcpubind=48,49,50,51 python /workspace/intelai_models/inference/infer_detections.py -g /in_graph/ssdmobilenet_fp32_pretrained_model_combinedNMS.pb -i 1000 -w 200 -a 4 -e 1 -b 1 >> ${OUTPUT_DIR}/ssd-mobilenet_fp32_inference_bs1_cores4_instance12.log 2>&1 & \
  OMP_NUM_THREADS=4 numactl --localalloc --physcpubind=52,53,54,55 python /workspace/intelai_models/inference/infer_detections.py -g /in_graph/ssdmobilenet_fp32_pretrained_model_combinedNMS.pb -i 1000 -w 200 -a 4 -e 1 -b 1 >> ${OUTPUT_DIR}/ssd-mobilenet_fp32_inference_bs1_cores4_instance13.log 2>&1 & \
  wait

  The following log files were saved to the output directory:
  ssd-mobilenet_fp32_inference_bs1_cores4_instance0.log
  ssd-mobilenet_fp32_inference_bs1_cores4_instance1.log
  ssd-mobilenet_fp32_inference_bs1_cores4_instance2.log
  ssd-mobilenet_fp32_inference_bs1_cores4_instance3.log
  ssd-mobilenet_fp32_inference_bs1_cores4_instance4.log
  ssd-mobilenet_fp32_inference_bs1_cores4_instance5.log
  ssd-mobilenet_fp32_inference_bs1_cores4_instance6.log
  ssd-mobilenet_fp32_inference_bs1_cores4_instance7.log
  ssd-mobilenet_fp32_inference_bs1_cores4_instance8.log
  ssd-mobilenet_fp32_inference_bs1_cores4_instance9.log
  ssd-mobilenet_fp32_inference_bs1_cores4_instance10.log
  ssd-mobilenet_fp32_inference_bs1_cores4_instance11.log
  ssd-mobilenet_fp32_inference_bs1_cores4_instance12.log
  ssd-mobilenet_fp32_inference_bs1_cores4_instance13.log

  A combined log file was saved to the output directory:
  ssd-mobilenet_fp32_inference_bs1_cores4_all_instances.log
  ```
  The following grep command can be used to get a summary of the total samples
  per second from the combined log file:
  ```
  grep 'Total samples/sec' ${OUTPUT_DIR}/ssd-mobilenet_fp32_inference_bs1_cores4_all_instances.log  | awk -F' ' '{sum+=$3;} END{print sum} '
  ```
