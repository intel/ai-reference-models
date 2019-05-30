# R-FCN (ResNet101)

This document has instructions for how to run R-FCN for the
following modes/precisions:
* [Int8 inference](#int8-inference-instructions)
* [FP32 inference](#fp32-inference-instructions)

Instructions and scripts for the R-FCN ResNet101 model training and inference
for other precisions are coming later.

## Int8 Inference Instructions

1. Clone the [tensorflow/models](https://github.com/tensorflow/models) and [cocodataset/cocoapi](https://github.com/cocodataset/cocoapi) repositories:

```
$ git clone https://github.com/tensorflow/models.git
$ cd models
$ git clone https://github.com/cocodataset/cocoapi.git

```

The TensorFlow models repo will be used for installing dependencies and running inference as well as
converting the coco dataset to the TF records format.

For the accuracy test, a modification is required in the cloned models repo until [this issue](https://github.com/tensorflow/models/issues/5411)
gets fixed in the TensorFlow repository.
This can be done either manually or using the command line as shown:

Open the file RFCN/models/research/object_detection/metrics/offline_eval_map_corloc.py,
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
      --train_image_dir="/home/<user>/coco/empty_dir" \
      --val_image_dir="/home/<user>/coco/val/val2017" \
      --test_image_dir="/home/<user>/coco/empty_dir" \
      --train_annotations_file="/home/<user>/coco/annotations/empty.json" \
      --val_annotations_file="/home/<user>/coco/annotations/instances_val2017.json" \
      --testdev_annotations_file="/home/<user>/coco/annotations/empty.json" \
      --output_dir="/home/<user>/coco/output"

$ ll /home/<user>/coco/output
total 1598276
-rw-rw-r--. 1 <user> <group>         0 Nov  2 21:46 coco_testdev.record
-rw-rw-r--. 1 <user> <group>         0 Nov  2 21:46 coco_train.record
-rw-rw-r--. 1 <user> <group> 818336740 Nov  2 21:46 coco_val.record

# Go back to the main models directory and get master code
$ cd /home/<user>/models
$ git checkout master
```

The `coco_val.record-00000-of-00001` file is what we will use in this inference example.

4. Download the pretrained model:
```
$ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/rfcn_resnet101_int8_coco_pretrained_model.pb
```

5. Clone the [intelai/models](https://github.com/intelai/models) repo
and then run the scripts for either batch/online inference performance or accuracy.

```
$ git clone https://github.com/IntelAI/models.git
$ cd models/benchmarks
```

Run for batch and online inference where the `--data-location`
is the path to the directory with the raw coco validation images:
```
python launch_benchmark.py \
    --model-name rfcn \
    --mode inference \
    --precision int8 \
    --framework tensorflow \
    --docker-image intelaipg/intel-optimized-tensorflow:PR25765-devel-mkl \
    --model-source-dir /home/<user>/tensorflow/models \
    --data-location /home/<user>/val/val2017 \
    --in-graph /home/<user>/rfcn_resnet101_int8_coco_pretrained_model.pb \
    --verbose \
    --benchmark-only \
    -- number_of_steps=500
```

Or for accuracy where the `--data-location` is the path the directory
where your `coco_val.record-00000-of-00001` file is located:
```
python launch_benchmark.py \
    --model-name rfcn \
    --mode inference \
    --precision int8 \
    --framework tensorflow \
    --docker-image intelaipg/intel-optimized-tensorflow:PR25765-devel-mkl \
    --model-source-dir /home/<user>/tensorflow/models \
    --data-location /home/<user>/coco/output/coco_val.record-00000-of-00001 \
    --in-graph /home/<user>/rfcn_resnet101_int8_coco_pretrained_model.pb \
    --accuracy-only \
    -- split="accuracy_message"
```

The docker image (`intelaipg/intel-optimized-tensorflow:PR25765-devel-mkl`)
used in the commands above were built using
[TensorFlow](git@github.com:tensorflow/tensorflow.git) master
([e889ea1](https://github.com/tensorflow/tensorflow/commit/e889ea1dd965c31c391106aa3518fc23d2689954)) and
[PR #25765](https://github.com/tensorflow/tensorflow/pull/25765).

Note that the `--verbose` or `--output-dir` flag can be added to any of the above commands
to get additional debug output or change the default output location.

7. Log files are located at the value of `--output-dir` (or
`models/benchmarks/common/tensorflow/logs` if no path has been specified):

Below is a sample log file tail when running for batch
and online inference:
```
Step 0: 10.6923000813 seconds
Step 10: 0.168856859207 seconds
...
Step 460: 0.181148052216 seconds
Step 470: 0.202737092972 seconds
Step 480: 0.117042064667 seconds
Step 490: 0.103501081467 seconds
Avg. Duration per Step:0.169812122345
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
Ran inference with batch size -1
Log location outside container: {--output-dir}/benchmark_rfcn_inference_int8_20190227_191959.log
```

And here is a sample log file tail when running for accuracy:
```
...
Accumulating evaluation results...
DONE (t=1.03s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.119
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.211
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.118
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.119
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.120
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.150
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.150
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.150
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
Ran inference with batch size -1
Log location outside container: {--output-dir}/benchmark_rfcn_inference_int8_20190227_194752.log
```


## Int8 Inference Instructions

1. Clone the [tensorflow/models](https://github.com/tensorflow/models) and [cocodataset/cocoapi](https://github.com/cocodataset/cocoapi) repositories:

```
$ git clone https://github.com/tensorflow/models.git
$ cd models
$ git clone https://github.com/cocodataset/cocoapi.git

```

The TensorFlow models repo will be used for installing dependencies and running inference as well as
converting the coco dataset to the TF records format.

For the accuracy test, a modification is required in the cloned models repo until [this issue](https://github.com/tensorflow/models/issues/5411)
gets fixed in the TensorFlow repository.
This can be done either manually or using the command line as shown:

Open the file RFCN/models/research/object_detection/metrics/offline_eval_map_corloc.py,
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
      --train_image_dir="/home/<user>/coco/empty_dir" \
      --val_image_dir="/home/<user>/coco/val/val2017" \
      --test_image_dir="/home/<user>/coco/empty_dir" \
      --train_annotations_file="/home/<user>/coco/annotations/empty.json" \
      --val_annotations_file="/home/<user>/coco/annotations/instances_val2017.json" \
      --testdev_annotations_file="/home/<user>/coco/annotations/empty.json" \
      --output_dir="/home/<user>/coco/output"

$ ll /home/<user>/coco/output
total 1598276
-rw-rw-r--. 1 <user> <group>         0 Nov  2 21:46 coco_testdev.record
-rw-rw-r--. 1 <user> <group>         0 Nov  2 21:46 coco_train.record
-rw-rw-r--. 1 <user> <group> 818336740 Nov  2 21:46 coco_val.record

# Go back to the main models directory and get master code
$ cd /home/<user>/models
$ git checkout master
```

The `coco_val.record-00000-of-00001` file is what we will use in this inference example.

4. Download the pretrained model:
```
$ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/rfcn_resnet101_int8_coco_pretrained_model.pb
```

5. Clone the [intelai/models](https://github.com/intelai/models) repo
and then run the benchmarking scripts for either benchmarking throughput
and latency or accuracy.

```
$ git clone https://github.com/IntelAI/models.git
$ cd models/benchmarks
```

Run benchmarking for throughput and latency where the `--data-location`
is the path to the directory with the raw coco validation images:
```
python launch_benchmark.py \
    --model-name rfcn \
    --mode inference \
    --precision int8 \
    --framework tensorflow \
    --docker-image intelaipg/intel-optimized-tensorflow:PR25765-devel-mkl \
    --model-source-dir /home/<user>/tensorflow/models \
    --data-location /home/<user>/val/val2017 \
    --in-graph /home/<user>/rfcn_resnet101_int8_coco_pretrained_model.pb \
    --verbose \
    --benchmark-only \
    -- number_of_steps=500
```

Or for accuracy where the `--data-location` is the path the directory
where your `coco_val.record-00000-of-00001` file is located:
```
python launch_benchmark.py \
    --model-name rfcn \
    --mode inference \
    --precision int8 \
    --framework tensorflow \
    --docker-image intelaipg/intel-optimized-tensorflow:PR25765-devel-mkl \
    --model-source-dir /home/<user>/tensorflow/models \
    --data-location /home/<user>/coco/output/coco_val.record-00000-of-00001 \
    --in-graph /home/<user>/rfcn_resnet101_int8_coco_pretrained_model.pb \
    --accuracy-only \
    -- split="accuracy_message"
```

The docker image (`intelaipg/intel-optimized-tensorflow:PR25765-devel-mkl`)
used in the commands above were built using
[TensorFlow](git@github.com:tensorflow/tensorflow.git) master
([e889ea1](https://github.com/tensorflow/tensorflow/commit/e889ea1dd965c31c391106aa3518fc23d2689954)) and
[PR #25765](https://github.com/tensorflow/tensorflow/pull/25765).

Note that the `--verbose` or `--output-dir` flag can be added to any of the above commands
to get additional debug output or change the default output location.

7. Log files are located at the value of `--output-dir` (or
`models/benchmarks/common/tensorflow/logs` if no path has been specified):

Below is a sample log file tail when running benchmarking for throughput
and latency:
```
Step 0: 10.6923000813 seconds
Step 10: 0.168856859207 seconds
...
Step 460: 0.181148052216 seconds
Step 470: 0.202737092972 seconds
Step 480: 0.117042064667 seconds
Step 490: 0.103501081467 seconds
Avg. Duration per Step:0.169812122345
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
Ran inference with batch size -1
Log location outside container: {--output-dir}/benchmark_rfcn_inference_int8_20190227_191959.log
```

And here is a sample log file tail when running for accuracy:
```
...
Accumulating evaluation results...
DONE (t=1.03s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.119
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.211
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.118
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.119
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.120
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.150
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.150
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.150
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
Ran inference with batch size -1
Log location outside container: {--output-dir}/benchmark_rfcn_inference_int8_20190227_194752.log
```


## FP32 Inference Instructions

1. Clone the `tensorflow/models` and `cocoapi` repositories:

```
$ git clone https://github.com/tensorflow/models.git
$ cd models
$ git clone https://github.com/cocodataset/cocoapi.git

```

The TensorFlow models repo will be used for running inference as well as
converting the coco dataset to the TF records format.

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
      --train_image_dir="/home/<user>/coco/empty_dir" \
      --val_image_dir="/home/<user>/coco/val/val2017" \
      --test_image_dir="/home/<user>/coco/empty_dir" \
      --train_annotations_file="/home/<user>/coco/annotations/empty.json" \
      --val_annotations_file="/home/<user>/coco/annotations/instances_val2017.json" \
      --testdev_annotations_file="/home/<user>/coco/annotations/empty.json" \
      --output_dir="/home/<user>/coco/output"

$ ll /home/<user>/coco/output
total 1598276
-rw-rw-r--. 1 <user> <group>         0 Nov  2 21:46 coco_testdev.record
-rw-rw-r--. 1 <user> <group>         0 Nov  2 21:46 coco_train.record
-rw-rw-r--. 1 <user> <group> 818336740 Nov  2 21:46 coco_val.record

# Go back to the main models directory and get master code
$ cd /home/<user>/models
$ git checkout master
```

The `coco_val.record` file is what we will use in this inference example.

4. <a name="download_fp32_pretrained_model"></a>Download and extract the pretrained model:

```
$ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/rfcn_resnet101_fp32_coco_pretrained_model.tar.gz
$ tar -xzvf rfcn_resnet101_fp32_coco_pretrained_model.tar.gz
```

5. Clone the [intelai/models](https://github.com/intelai/models) repo.
This repo has the launch script for running the model.

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

6. Run the `launch_benchmark.py` script from the 
[intelai/models](https://github.com/intelai/models) repo
, with the appropriate parameters including: the 
`coco_val.record` data location (from step 3), the pre-trained model
`pipeline.config` file and the checkpoint location (from step 4), and the
location of your `tensorflow/models` clone (from step 1).

Run for batch and online inference:
```
$ cd /home/<user>/models/benchmarks

$ python launch_benchmark.py \
    --data-location /home/<user>/coco/output/ \
    --model-source-dir /home/<user>/tensorflow/models \
    --model-name rfcn \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --socket-id 0 \
    --checkpoint /home/<user>/rfcn_resnet101_fp32_coco \
    --docker-image intelaipg/intel-optimized-tensorflow:latest-devel-mkl \
    -- config_file=rfcn_pipeline.config
```

Or for accuracy where the `--data-location` is the path the directory
where your `coco_val.record` file is located and the `--in-graph` is
the pre-trained graph located in the pre-trained model directory (from step 4):
```
python launch_benchmark.py \
    --model-name rfcn \
    --mode inference \
    --precision fp32 \
    --framework tensorflow \
    --docker-image intelaipg/intel-optimized-tensorflow:latest-devel-mkl \
    --model-source-dir /home/<user>/tensorflow/models \
    --data-location /home/<user>/coco/output/coco_val.record \
    --in-graph /home/<user>/rfcn_resnet101_fp32_coco/frozen_inference_graph.pb  \
    --accuracy-only \
    -- split="accuracy_message"
```

7. Log files are located at the value of `--output-dir` (or
`models/benchmarks/common/tensorflow/logs` if no path has been specified):

Below is a sample log file tail when running for batch and
online inference:

```
Average time per step: 0.262 sec
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
Received these standard args: Namespace(accuracy_only=False, batch_size=1, benchmark_only=False, checkpoint='/checkpoints', data_location='/dataset', framework='tensorflow', input_graph=None, intelai_models='/workspace/intelai_models', mode='inference', model_args=[], model_name='rfcn', model_source_dir='/workspace/models', num_cores=-1, num_inter_threads=2, num_intra_threads=56, precision='fp32, socket_id=0, use_case='object_detection', verbose=True)
Received these custom args: ['--config_file=rfcn_pipeline.config']
Run model here.
current directory: /workspace/models/research
Running: numactl --cpunodebind=0 --membind=0 python /workspace/intelai_models/inference/fp32/eval.py --inter_op 1 --intra_op 28 --omp 28 --pipeline_config_path /checkpoints/rfcn_pipeline.config --checkpoint_dir /checkpoints --eval_dir /workspace/models/research/object_detection/models/rfcn/eval  --logtostderr  --blocktime=0  --run_once=True 
PYTHONPATH: :/workspace/intelai_models:/workspace/models/research:/workspace/models/research/slim:/workspace/models
RUNCMD: python common/tensorflow/run_tf_benchmark.py --framework=tensorflow --use-case=object_detection --model-name=rfcn --precision=fp32 --mode=inference --model-source-dir=/workspace/models --intelai-models=/workspace/intelai_models --num-cores=-1 --batch-size=1 --data-location=/dataset --socket-id 0 --verbose --checkpoint=/checkpoints         --config_file=rfcn_pipeline.config
Batch Size: 1
Ran inference with batch size 1
Log location outside container: {--output-dir value}/benchmark_rfcn_inference.log
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
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
Ran inference with batch size 1
Log location outside container: {--output-dir value}/benchmark_rfcn_inference_fp32_20181221_211905.log
```
