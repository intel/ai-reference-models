<!-- 50. Launch benchmark instructions -->
Once your environment is setup, navigate to the `benchmarks` directory of
the model zoo and set environment variables pointing to the directory for the
dataset, pretrained model frozen graph, the TensorFlow models repo, and an output
directory where log files will be written.

```
# cd to the benchmarks directory in the model zoo
cd benchmarks

export DATASET_DIR=<path to the dataset (raw images for inference or the TF records file for accuracy)>
export PRETRAINED_MODEL=<path to the pretrained model frozen graph file>
export TF_MODELS_DIR=<path to your clone of the TensorFlow models repo>
export OUTPUT_DIR=<directory where log files will be saved>
```

The command below runs batch and online inference. Note that the
`--data-location ${DATASET_DIR}` should point to the raw COCO dataset images
(for example `DATASET_DIR=/home/<user>/coco_dataset/val2017`).
```
python launch_benchmark.py \
    --model-name rfcn \
    --mode inference \
    --precision int8 \
    --framework tensorflow \
    --docker-image <docker image> \
    --model-source-dir ${TF_MODELS_DIR} \
    --data-location ${DATASET_DIR} \
    --in-graph ${PRETRAINED_MODEL} \
    --benchmark-only \
    --output-dir ${OUTPUT_DIR} \
    -- number_of_steps=500
```

Or for accuracy testing, use the command below and set the `--data-location ${DATASET_DIR}`
to the path the TF record file (for example `DATASET_DIR=/home/<user>/coco_output/coco_val.record`).
```
python launch_benchmark.py \
    --model-name rfcn \
    --mode inference \
    --precision int8 \
    --framework tensorflow \
    --docker-image <docker image> \
    --model-source-dir ${TF_MODELS_DIR} \
    --data-location ${DATASET_DIR} \
    --in-graph ${PRETRAINED_MODEL} \
    --accuracy-only \
    --output-dir ${OUTPUT_DIR} \
    -- split="accuracy_message"
```

Note that the `--verbose` flag can be added to any of the above commands
to get additional debug output.

Log files are located at the `${OUTPUT_DIR}` path.

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
Log location outside container: ${OUTPUT_DIR}/benchmark_rfcn_inference_int8_20190416_182445.log
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
Log location outside container: ${OUTPUT_DIR}/benchmark_rfcn_inference_int8_20190227_194752.log
```
