# R-FCN (ResNet101)

This document has instructions for how to run R-FCN for the
following modes/platforms:
* [Int8 inference](#int8-inference-instructions)
* [FP32 inference](#fp32-inference-instructions)

Benchmarking instructions and scripts for the R-FCN ResNet101 model training and inference
other platforms are coming later.

## Int8 Inference Instructions

1. Clone the [tensorflow/models](https://github.com/tensorflow/models) and [cocodataset/cocoapi](https://github.com/cocodataset/cocoapi) repositories:

```
$ git clone git@github.com:tensorflow/models.git
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

2. Build a docker image using master of the official
[TensorFlow](https://github.com/tensorflow/tensorflow) repository with
`--config=mkl`. More instructions on
[how to build from source](https://software.intel.com/en-us/articles/intel-optimization-for-tensorflow-installation-guide#inpage-nav-5).


3.  Download the 2017 validation
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

4. Now that you have the raw COCO dataset, we need to convert it to the
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

The `coco_val.record-00000-of-00001` file is what we will use in this inference example.

5. download the pre-trained model:
```
$ wget https://storage.cloud.google.com/intel-optimized-tensorflow/models/rfcn_resnet101_int8_coco_pretrained_model.pb
```

6. Clone the [intelai/models](https://github.com/intelai/models) repo
and then run the benchmarking scripts for either benchmarking throughput
and latency or accuracy.

```
$ git clone git@github.com:IntelAI/models.git

$ cd benchmarks
```

Run benchmarking for throughput and latency where the `--data-location`
is the path to the directory with the raw coco validation images:
```
python launch_benchmark.py \
    --model-name rfcn \
    --mode inference \
    --platform int8 \
    --framework tensorflow \
    --docker-image your-docker-image \
    --model-source-dir /home/myuser/tensorflow/models \
    --data-location /home/myuser/val/val2017 \
    --in-graph /home/myuser/rfcn_resnet101_int8_coco_pretrained_model.pb \
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
    --platform int8 \
    --framework tensorflow \
    --docker-image your-docker-image \
    --model-source-dir /home/myuser/tensorflow/models \
    --data-location /home/myuser/coco/output/coco_val.record-00000-of-00001 \
    --in-graph /home/myuser/rfcn_resnet101_int8_coco_pretrained_model.pb \
    --verbose \
    --accuracy-only \
    -- split="accuracy_message"
```

7. Log files are located at:
`intelai/models/benchmarks/common/tensorflow/logs`.

Below is a sample log file tail when running benchmarking for throughput
and latency:
```
Step 0: 10.2767250538 seconds
Step 10: 0.123119115829 seconds
...
Step 450: 0.0954110622406 seconds
Step 460: 0.0991611480713 seconds
Step 470: 0.0990519523621 seconds
Step 480: 0.0993700027466 seconds
Step 490: 0.117154121399 seconds
Avg. Duration per Step:0.122102419376
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
Received these standard args: Namespace(accuracy_only=False, batch_size=-1, benchmark_only=True, checkpoint=None, data_location='/dataset', evaluate_tensor=None, framework='tensorflow', input_graph='/in_graph/rfcn_resnet101_int8_coco_pretrained_model.pb', intelai_models='/workspace/intelai_models', mode='inference', model_args=[], model_name='rfcn', model_source_dir='/workspace/models', num_cores=-1, num_inter_threads=2, num_intra_threads=56, number_of_steps=500, platform='int8', print_accuracy=False, single_socket=False, socket_id=0, split=None, timeline=None, use_case='object_detection', verbose=True, visualize=False)
Received these custom args: ['--number_of_steps=500']
Current directory: /workspace/models/research
Running: /usr/bin/python /workspace/intelai_models/inference/int8/run_rfcn_inference.py -m /workspace/models -g /in_graph/rfcn_resnet101_int8_coco_pretrained_model.pb -x 500 -d /dataset
PYTHONPATH: :/workspace/intelai_models:/workspace/models/research:/workspace/models/research/slim:/workspace/models
RUNCMD: python common/tensorflow/run_tf_benchmark.py --framework=tensorflow --use-case=object_detection --model-name=rfcn --platform=int8 --mode=inference --model-source-dir=/workspace/models --intelai-models=/workspace/intelai_models --num-cores=-1 --batch-size=-1  --verbose --in-graph=/in_graph/rfcn_resnet101_int8_coco_pretrained_model.pb --data-location=/dataset      --benchmark-only --number_of_steps=500  
Batch Size: -1
Ran inference with batch size -1
Log location outside container: /home/myuser/intelai/benchmarks/common/tensorflow/logs/benchmark_rfcn_inference_int8_20181206_204825.log

```

And here is a sample log file tail when running for accuracy:
```
DONE (t=0.01s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
Received these standard args: Namespace(accuracy_only=True, batch_size=-1, benchmark_only=False, checkpoint=None, data_location='/dataset', evaluate_tensor=None, framework='tensorflow', input_graph='/in_graph/rfcn_resnet101_int8_coco_pretrained_model.pb', intelai_models='/workspace/intelai_models', mode='inference', model_args=[], model_name='rfcn', model_source_dir='/workspace/models', num_cores=-1, num_inter_threads=2, num_intra_threads=56, number_of_steps=None, platform='int8', print_accuracy=False, single_socket=False, socket_id=0, split='accuracy_message', timeline=None, use_case='object_detection', verbose=True, visualize=False)
Received these custom args: ['--split=accuracy_message']
Current directory: /workspace/models/research
Running: FROZEN_GRAPH=/in_graph/rfcn_resnet101_int8_coco_pretrained_model.pb TF_RECORD_FILE=/dataset SPLIT=accuracy_message TF_MODELS_ROOT=/workspace/models /workspace/intelai_models/inference/int8/coco_mAP.sh
PYTHONPATH: :/workspace/intelai_models:/workspace/models/research:/workspace/models/research/slim:/workspace/models
RUNCMD: python common/tensorflow/run_tf_benchmark.py --framework=tensorflow --use-case=object_detection --model-name=rfcn --platform=int8 --mode=inference --model-source-dir=/workspace/models --intelai-models=/workspace/intelai_models --num-cores=-1 --batch-size=-1  --verbose --in-graph=/in_graph/rfcn_resnet101_int8_coco_pretrained_model.pb --data-location=/dataset     --accuracy-only   --split=accuracy_message 
Batch Size: -1
Ran inference with batch size -1
Log location outside container: /home/myuser/intelai/benchmarks/common/tensorflow/logs/benchmark_rfcn_inference_int8_20181206_225054.log
```


## FP32 Inference Instructions

1. Clone the `tensorflow/models` and `cocoapi` repositories:

```
$ git clone git@github.com:tensorflow/models.git
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

4. Download the pre-trained model rfcn_resnet101_fp32_coco_pretrained_model.tar.gz.
The pre-trained model includes the checkpoint files and the R-FCN ResNet101 model `rfcn_pipeline.config`.
Extract and check out its contents as shown:
```
$ wget https://storage.cloud.google.com/intel-optimized-tensorflow/models/rfcn_resnet101_fp32_coco_pretrained_model.tar.gz
$ tar -xzvf rfcn_resnet101_fp32_coco_pretrained_model.tar.gz
$ ls -l rfcn_resnet101_fp32_coco
total 374848
-rw-r--r--  1 myuser  myuser         77 Nov 12 22:33 checkpoint
-rw-r--r--  1 myuser  myuser  176914228 Nov 12 22:33 model.ckpt.data-00000-of-00001
-rw-r--r--  1 myuser  myuser      14460 Nov 12 22:33 model.ckpt.index
-rw-r--r--  1 myuser  myuser    5675175 Nov 12 22:33 model.ckpt.meta
-rwxr--r--  1 myuser  myuser       5056 Nov 12 22:33 mscoco_label_map.pbtxt
-rwxr-xr-x  1 myuser  myuser       3244 Nov 12 22:33 rfcn_pipeline.config
drwxr-xr-x  4 myuser  myuser        128 Nov 12 22:30 saved_model

```
Make sure that the `eval_input_reader` section in the `rfcn_pipeline.config` file has the mounted 
`coco_val.record` data and pre-trained model `mscoco_label_map.pbtxt` location.

5. Clone the [intelai/models](https://github.com/intelai/models) repo.
This repo has the launch script for running benchmarking.

```
$ git clone git@github.com:IntelAI/models.git
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

```
$ cd /home/myuser/models/benchmarks

$ python launch_benchmark.py \
    --data-location /home/myuser/coco/output/ \
    --model-source-dir /home/myuser/tensorflow/models \
    --model-name rfcn \
    --framework tensorflow \
    --platform fp32 \
    --mode inference \
    --checkpoint /home/myuser/rfcn_resnet101_fp32_coco \
    --docker-image intelaipg/intel-optimized-tensorflow:latest-devel-mkl \
    -- config-file=rfcn_pipeline.config
```

7. The log file is saved to:
`models/benchmarks/common/tensorflow/logs`

The tail of the log output when the benchmarking completes should look
something like this:

```
Average time per step: 0.262 sec
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
Received these standard args: Namespace(accuracy_only=False, batch_size=1, benchmark_only=False, checkpoint='/checkpoints', data_location='/dataset', framework='tensorflow', input_graph=None, intelai_models='/workspace/intelai_models', mode='inference', model_args=[], model_name='rfcn', model_source_dir='/workspace/models', num_cores=-1, num_inter_threads=2, num_intra_threads=56, platform='fp32', single_socket=True, socket_id=0, use_case='object_detection', verbose=True)
Received these custom args: ['--config_file=rfcn_pipeline.config']
Run model here.
current directory: /workspace/models/research
Running: numactl --cpunodebind=0 --membind=0 python /workspace/intelai_models/inference/fp32/eval.py --inter_op 1 --intra_op 28 --omp 28 --pipeline_config_path /checkpoints/rfcn_pipeline.config --checkpoint_dir /checkpoints --eval_dir /workspace/models/research/object_detection/models/rfcn/eval  --logtostderr  --blocktime=0  --run_once=True 
PYTHONPATH: :/workspace/intelai_models:/workspace/models/research:/workspace/models/research/slim:/workspace/models
RUNCMD: python common/tensorflow/run_tf_benchmark.py --framework=tensorflow --use-case=object_detection --model-name=rfcn --platform=fp32 --mode=inference --model-source-dir=/workspace/models --intelai-models=/workspace/intelai_models --num-cores=-1 --batch-size=1 --data-location=/dataset --single-socket --verbose --checkpoint=/checkpoints         --config_file=rfcn_pipeline.config 
Batch Size: 1
Ran inference with batch size 1
Log location outside container: /home/myuser/models/benchmarks/common/tensorflow/logs/benchmark_rfcn_inference.log
```

