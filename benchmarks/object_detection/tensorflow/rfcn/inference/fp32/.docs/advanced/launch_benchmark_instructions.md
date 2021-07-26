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

The command below runs batch and online inference. Note that the
`--data-location ${DATASET_DIR}` should point to the raw COCO dataset images
(for example: `DATASET_DIR=/home/<user>/coco_dataset/val2017`).
```
python launch_benchmark.py \
    --model-name rfcn \
    --mode inference \
    --precision fp32 \
    --framework tensorflow \
    --docker-image <docker image> \
    --model-source-dir ${TF_MODELS_DIR} \
    --data-location ${DATASET_DIR} \
    --in-graph ${PRETRAINED_MODEL} \
    --output-dir ${OUTPUT_DIR} \
    --benchmark-only \
    -- number_of_steps=500
```

Or for accuracy testing, use the command below and set the `--data-location ${DATASET_DIR}`
to the path to the COCO validation TF records file (for example:
`DATASET_DIR=/home/<user>/coco_output/coco_val.record`).
```
python launch_benchmark.py \
    --model-name rfcn \
    --mode inference \
    --precision fp32 \
    --framework tensorflow \
    --docker-image <docker image> \
    --model-source-dir ${TF_MODELS_DIR} \
    --data-location ${DATASET_DIR} \
    --in-graph ${PRETRAINED_MODEL} \
    --output-dir ${OUTPUT_DIR} \
    --accuracy-only \
    -- split="accuracy_message"
```

Note that the `--verbose` flag can be added to any of the above commands
to get additional debug output .

Log files are located at the value of `$OUTPUT_DIR`. Below is a sample log
file tail when running for batch and online inference:
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
Log location outside container: ${OUTPUT_DIR}/benchmark_rfcn_inference.log
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
Log location outside container: ${OUTPUT_DIR}/benchmark_rfcn_inference_fp32_20181221_211905.log
```
