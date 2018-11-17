# Fast R-CNN (ResNet50)

This document has instructions for how to run FastRCNN for the
following modes/platforms:
* [FP32 inference](#fp32-inference-instructions)

Benchmarking instructions and scripts for the Fast R-CNN ResNet50 model training and inference
other platforms are coming later.

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

4. Download the pre-trained model fast_rcnn_resnet50_fp32_coco_pretrained_model.tar.gz.
The pre-trained model includes the checkpoint files and the Fast R-CNN ResNet50 model `pipeline.config`.
Extract and check out its contents as shown:
```
$ wget https://storage.cloud.google.com/intel-optimized-tensorflow/models/fast_rcnn_resnet50_fp32_coco_pretrained_model.tar.gz
$ tar -xzvf fast_rcnn_resnet50_fp32_coco_pretrained_model.tar.gz
$ ls -l fast_rcnn_resnet50_fp32_coco
total 374848
-rw-r--r--  1 myuser  myuser         77 Nov 12 22:33 checkpoint
-rw-r--r--  1 myuser  myuser  176914228 Nov 12 22:33 model.ckpt.data-00000-of-00001
-rw-r--r--  1 myuser  myuser      14460 Nov 12 22:33 model.ckpt.index
-rw-r--r--  1 myuser  myuser    5675175 Nov 12 22:33 model.ckpt.meta
-rwxr--r--  1 myuser  myuser       5056 Nov 12 22:33 mscoco_label_map.pbtxt
-rwxr-xr-x  1 myuser  myuser       3244 Nov 12 22:33 pipeline.config
drwxr-xr-x  4 myuser  myuser        128 Nov 12 22:30 saved_model

```
Make sure that the `eval_input_reader` section in the `pipeline.config` file has the mounted 
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
    --model-name fastrcnn \
    --framework tensorflow \
    --platform fp32 \
    --mode inference \
    --checkpoint /home/myuser/fast_rcnn_resnet50_fp32_coco \
    --docker-image intelaipg/intel-optimized-tensorflow:latest-devel-mkl \
    -- config-file=pipeline.config
```

7. The log file is saved to:
models/benchmarks/common/tensorflow/logs/benchmark_fastrcnn_inference.log

The tail of the log output when the benchmarking completes should look
something like this:

```
Time spent : 167.353 seconds.
Time spent per BATCH: 0.167 seconds.
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
Received these standard args: Namespace(accuracy_only=False, batch_size=1, benchmark_only=False, checkpoint='/checkpoints', data_location='/dataset', framework='tensorflow', input_graph=None, intelai_models='/workspace/intelai_models', mode='inference', model_args=[], model_name='fastrcnn', model_source_dir='/workspace/models', num_cores=-1, num_inter_threads=2, num_intra_threads=56, platform='fp32', single_socket=True, socket_id=0, use_case='object_detection', verbose=True)
Received these custom args: ['--config_file=pipeline.config']
Run model here.
current directory: /workspace/models/research
Running: numactl --cpunodebind=0 --membind=0 python /workspace/intelai_models/inference/fp32/eval.py --num_inter_threads 1 --num_intra_threads 28 --pipeline_config_path /checkpoints/pipeline.config --checkpoint_dir /checkpoints --eval_dir /workspace/models/research/object_detection/log/eval
PYTHONPATH: :/workspace/intelai_models:/workspace/models/research:/workspace/models/research/slim:/workspace/models
RUNCMD: python common/tensorflow/run_tf_benchmark.py --framework=tensorflow --use-case=object_detection --model-name=fastrcnn --platform=fp32 --mode=inference --model-source-dir=/workspace/models --intelai-models=/workspace/intelai_models --num-cores=-1 --batch-size=1 --data-location=/dataset --single-socket --verbose --checkpoint=/checkpoints         --config_file=pipeline.config 
Batch Size: 1
Ran inference with batch size 1
Log location outside container: /home/myuser/models/benchmarks/common/tensorflow/logs/benchmark_fastrcnn_inference.log
```
