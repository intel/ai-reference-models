<!-- 50. Launch benchmark instructions -->
Once your environment is setup, navigate to the `benchmarks` directory of
the model zoo and set environment variables pointing to the directory for the
dataset, pretrained model frozen graph, the TensorFlow models repo, and an output
directory where log files will be written.

```
# cd to the benchmarks directory in the model zoo
cd benchmarks

export DATASET_DIR=<path to the dataset>
export PRETRAINED_MODEL=<path to the pretrained model frozen graph file>
export TF_MODELS_DIR=<path to your clone of the TensorFlow models repo>
export OUTPUT_DIR=<directory where log files will be saved>
```
Run batch and online inference using the following command.
```
python launch_benchmark.py \
 --data-location ${DATASET_DIR} \
 --model-source-dir ${TF_MODELS_DIR} \
 --model-name faster_rcnn \
 --framework tensorflow \
 --precision int8 \
 --mode inference \
 --socket-id 0 \
 --in-graph ${PRETRAINED_MODEL} \
 --docker-image <docker image> \
 --output-dir ${OUTPUT_DIR} \
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
 --docker-image <docker image> \
 --model-source-dir ${TF_MODELS_DIR} \
 --data-location ${DATASET_DIR}/coco_val.record \
 --in-graph ${PRETRAINED_MODEL} \
 --output-dir ${OUTPUT_DIR} \
 --accuracy-only
```

Output files and logs are saved to the `${OUTPUT_DIR}` directory.

Below is a sample log file tail when running for batch and online inference:
```
Step 4970: ... seconds
Step 4980: ... seconds
Step 4990: ... seconds
Avg. Duration per Step: ...
Log location outside container: ${OUTPUT_DIR}/benchmark_faster_rcnn_inference_int8_20190117_232539.log
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
Log location outside container: ${OUTPUT_DIR}/benchmark_faster_rcnn_inference_int8_20190117_231937.log
```