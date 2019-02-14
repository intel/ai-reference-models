# Fast R-CNN (ResNet50)

This document has instructions for how to run FastRCNN for the
following modes/precisions:
* [FP32 inference](#fp32-inference-instructions)
* [Int8 inference](#int8-inference-instructions)

Benchmarking instructions and scripts for the Fast R-CNN ResNet50 model training and inference
other precisions are coming later.

## FP32 Inference Instructions

1. Clone the `tensorflow/models` and `cocoapi` repositories:

```
$ git clone https://github.com/tensorflow/models.git
$ cd models
$ git clone https://github.com/cocodataset/cocoapi.git

```

The TensorFlow models repo will be used for running inference as well as
converting the coco dataset to the TF records format.

For the accuracy test, a modification is required in the cloned models repo until [this issue](https://github.com/tensorflow/models/issues/5411)
gets fixed in the TensorFlow repository.
This can be done either manually or using the command line as shown:

Open the file models/research/object_detection/metrics/offline_eval_map_corloc.py,
then apply the following fixes:
Line 162: change `configs['eval_input_config']` to `configs['eval_input_configs']`
Line 91, 92, and 95: change `input_config` to `input_config[0]`

Or using the command line:
```
cd models/research/object_detection
chmod 777 metrics
cd "metrics"
chmod 777 offline_eval_map_corloc.py
sed -i.bak 162s/eval_input_config/eval_input_configs/ offline_eval_map_corloc.py
sed -i.bak 91s/input_config/input_config[0]/ offline_eval_map_corloc.py
sed -i.bak 92s/input_config/input_config[0]/ offline_eval_map_corloc.py
sed -i.bak 95s/input_config/input_config[0]/ offline_eval_map_corloc.py

```

2.  Download the 2017 validation
[COCO dataset](http://cocodataset.org/#home) and annotations:

```
$ mkdir val
$ cd val
$ wget http://images.cocodataset.org/zips/val2017.zip
$ unzip val2017.zip
$ cd ..

$ mkdir annotations
$ cd annotations
$ wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
$ unzip annotations_trainval2017.zip
$ cd ..
```

Since we are only using the validation dataset in this example, we will
create an empty directory and empty annotations json file to pass as the
train and test directories in the next step.

```
$ mkdir empty_dir

$ cd annotations
$ echo "{ \"images\": {}, \"categories\": {}}" > empty.json
$ cd ..
```

3. Now that you have the raw COCO dataset, we need to convert it to the
TF records format in order to use it with the inference script.  We will
do this by running the `create_coco_tf_record.py` file in the TensorFlow
models repo.

Follow the steps below to navigate to the proper directory and point the
script to the raw COCO dataset files that you have downloaded in step 2.
The `--output_dir` is the location where the TF record files will be
located after the script has completed.

```

# We are going to use an older version of the conversion script to checkout the git commit
$ cd models
$ git checkout 7a9934df2afdf95be9405b4e9f1f2480d748dc40

$ cd research/object_detection/dataset_tools/
$ python create_coco_tf_record.py --logtostderr \
      --train_image_dir="/home/myuser/coco/empty_dir" \
      --val_image_dir="/home/myuser/coco/val/val2017" \
      --test_image_dir="/home/myuser/coco/empty_dir" \
      --train_annotations_file="/home/myuser/coco/annotations/empty.json" \
      --val_annotations_file="/home/myuser/coco/annotations/instances_val2017.json" \
      --testdev_annotations_file="/home/myuser/coco/annotations/empty.json" \
      --output_dir="/home/myuser/coco/output"

$ ll /home/myuser/coco/output
total 1598276
-rw-rw-r--. 1 myuser myuser         0 Nov  2 21:46 coco_testdev.record
-rw-rw-r--. 1 myuser myuser         0 Nov  2 21:46 coco_train.record
-rw-rw-r--. 1 myuser myuser 818336740 Nov  2 21:46 coco_val.record

# Go back to the main models directory and get master code
$ cd /home/myuser/models
$ git checkout master
```

The `coco_val.record` file is what we will use in this inference example.

4. A link to download the pre-trained model is coming soon.

5. Clone the [intelai/models](https://github.com/intelai/models) repo.
This repo has the launch script for running benchmarking.

```
$ git clone https://github.com/IntelAI/models.git
Cloning into 'models'...
remote: Enumerating objects: 11, done.
remote: Counting objects: 100% (11/11), done.
remote: Compressing objects: 100% (7/7), done.
remote: Total 11 (delta 3), reused 4 (delta 0), pack-reused 0
Receiving objects: 100% (11/11), done.
Resolving deltas: 100% (3/3), done.
```

6. Run the `launch_benchmark.py` script from the intelai/models repo
, with the appropriate parameters including: the 
`coco_val.record` data location (from step 3), the pre-trained model
`pipeline.config` file and the checkpoint location (from step 4, and the
location of your `tensorflow/models` clone (from step 1).

Run benchmarking for throughput and latency:
```
$ cd /home/myuser/models/benchmarks

$ python launch_benchmark.py \
    --data-location /home/myuser/coco/output/ \
    --model-source-dir /home/myuser/tensorflow/models \
    --model-name fastrcnn \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --socket-id 0 \
    --checkpoint /home/myuser/fast_rcnn_resnet50_fp32_coco \
    --docker-image intelaipg/intel-optimized-tensorflow:latest-devel-mkl \
    -- config_file=pipeline.config
```

Or for accuracy where the `--data-location` is the path the directory
where your `coco_val.record` file is located and the `--in-graph` is
the pre-trained graph located in the pre-trained model directory (from step 4):
```
python launch_benchmark.py \
    --model-name fastrcnn \
    --mode inference \
    --precision fp32 \
    --framework tensorflow \
    --docker-image intelaipg/intel-optimized-tensorflow:latest-devel-mkl \
    --model-source-dir /home/myuser/tensorflow/models \
    --data-location /home/myuser/coco/output \
    --in-graph /home/myuser/fast_rcnn_resnet50_fp32_coco/frozen_inference_graph.pb  \
    --accuracy-only
```

7. The log file is saved to the value of `--output-dir`.

Below is a sample log file tail when running benchmarking for throughput
and latency:

```
Time spent : 167.353 seconds.
Time spent per BATCH: 0.167 seconds.
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
Received these standard args: Namespace(accuracy_only=False, batch_size=1, benchmark_only=False, checkpoint='/checkpoints', data_location='/dataset', framework='tensorflow', input_graph=None, intelai_models='/workspace/intelai_models', mode='inference', model_args=[], model_name='fastrcnn', model_source_dir='/workspace/models', num_cores=-1, num_inter_threads=2, num_intra_threads=56, precision='fp32', socket_id=0, use_case='object_detection', verbose=True)
Received these custom args: ['--config_file=pipeline.config']
Run model here.
current directory: /workspace/models/research
Running: numactl --cpunodebind=0 --membind=0 python /workspace/intelai_models/inference/fp32/eval.py --num_inter_threads 1 --num_intra_threads 28 --pipeline_config_path /checkpoints/pipeline.config --checkpoint_dir /checkpoints --eval_dir /workspace/models/research/object_detection/log/eval
PYTHONPATH: :/workspace/intelai_models:/workspace/models/research:/workspace/models/research/slim:/workspace/models
RUNCMD: python common/tensorflow/run_tf_benchmark.py --framework=tensorflow --use-case=object_detection --model-name=fastrcnn --precision=fp32 --mode=inference --model-source-dir=/workspace/models --intelai-models=/workspace/intelai_models --num-cores=-1 --batch-size=1 --data-location=/dataset --socket-id 0 --verbose --checkpoint=/checkpoints         --config_file=pipeline.config
Batch Size: 1
Ran inference with batch size 1
Log location outside container: {--output-dir value}/benchmark_fastrcnn_inference.log
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
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
Ran inference with batch size 1
Log location outside container: {--output-dir value}/benchmark_fastrcnn_inference_fp32_20190114_205714.log
```

## Int8 Inference Instructions

1. Please follow step 1, 2 and 3 of Fast R-CNN FP32 instructions written above.

2. A link to download the pre-trained model is coming soon.

3. Clone the [intelai/models](https://github.com/intelai/models) repo.
This repo has the launch script for running benchmarking.

```
$ git clone https://github.com/IntelAI/models.git
Cloning into 'models'...
remote: Enumerating objects: 11, done.
remote: Counting objects: 100% (11/11), done.
remote: Compressing objects: 100% (7/7), done.
remote: Total 11 (delta 3), reused 4 (delta 0), pack-reused 0
Receiving objects: 100% (11/11), done.
Resolving deltas: 100% (3/3), done.
```

4. Run the `launch_benchmark.py` script from the intelai/models repo,
with the appropriate parameters. To run on single socket use `--socket_id` switch,
by default it will be using all available sockets. Optional parameter `number_of_steps`
(default value = 5000) can be added at the end of command after `--` as shown below:

Run benchmarking for throughput and latency:
```
$ cd /home/myuser/models/benchmarks

$ python launch_benchmark.py \
    --data-location /home/myuser/coco/output/ \
    --model-source-dir /home/myuser/tensorflow/models \
    --model-name fastrcnn \
    --framework tensorflow \
    --precision int8 \
    --mode inference \
    --socket-id 0 \
    --in-graph /home/myuser/pretrained_int8_fastrcnn_model.pb \
    --docker-image tf_int8_docker_image
    --benchmark-only
    -- number_of_steps=5000
```

Or for accuracy where the `--data-location` is the path the directory
where your `coco_val.record` file is located and the `--in-graph` is
the pre-trained graph model:
```
python launch_benchmark.py \
    --model-name fastrcnn \
    --mode inference \
    --precision int8 \
    --framework tensorflow \
    --socket-id 0 \
    --docker-image tf_int8_docker_image \
    --model-source-dir /home/myuser/tensorflow/models \
    --data-location /home/myuser/coco_dataset/coco_val.record \
    --in-graph /home/myuser/pretrained_int8_fastrcnn_model.pb  \
    --accuracy-only
```

5. The log file is saved to the value of `--output-dir`.

Below is a sample log file tail when running benchmarking for throughput
and latency:

```
Step 4950: 0.0722849369049 seconds
Step 4960: 0.0763049125671 seconds
Step 4970: 0.070191860199 seconds
Step 4980: 0.0755469799042 seconds
Step 4990: 0.0742928981781 seconds
Avg. Duration per Step:0.0760930150986
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
Ran inference with batch size -1
Log location outside container: {--output-dir value}/benchmark_fastrcnn_inference_int8_20190117_232539.log
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
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
Ran inference with batch size -1
Log location outside container: {--output-dir value}/benchmark_fastrcnn_inference_int8_20190117_231937.log
```

