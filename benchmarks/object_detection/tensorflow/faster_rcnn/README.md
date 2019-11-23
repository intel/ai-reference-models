# Faster R-CNN (ResNet50)

This document has instructions for how to run Faster R-CNN for the
following modes/precisions:
* [FP32 inference](#fp32-inference-instructions)

Instructions and scripts for the Faster R-CNN ResNet50 model training and inference
for other precisions are coming later.

## FP32 Inference Instructions

1. Store the path to the current directory:
```
$ MODEL_WORK_DIR=${MODEL_WORK_DIR:=`pwd`}
$ pushd $MODEL_WORK_DIR
```

2. Clone the `tensorflow/models` and `cocoapi` repositories:

```
$ git clone https://github.com/tensorflow/models.git tf_models
$ cd tf_models
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
$ cd $MODEL_WORK_DIR/tf_models/research/object_detection
$ chmod 777 metrics
$ cd "metrics"
$ chmod 777 offline_eval_map_corloc.py
$ sed -i.bak 162s/eval_input_config/eval_input_configs/ offline_eval_map_corloc.py
$ sed -i.bak 91s/input_config/input_config[0]/ offline_eval_map_corloc.py
$ sed -i.bak 92s/input_config/input_config[0]/ offline_eval_map_corloc.py
$ sed -i.bak 95s/input_config/input_config[0]/ offline_eval_map_corloc.py

```

3.  Download and unzip the 2017 validation
[COCO dataset](http://cocodataset.org/#home) images:

```
$ cd $MODEL_WORK_DIR
$ mkdir val
$ cd val
$ wget http://images.cocodataset.org/zips/val2017.zip
$ unzip val2017.zip
```

4. Download and unzip the coco dataset annotations file:
```
$ cd $MODEL_WORK_DIR
$ mkdir annotations
$ cd annotations
$ wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
$ unzip annotations_trainval2017.zip
```

Since we are only using the validation dataset in this example, we will
create an empty directory and empty annotations json file to pass as the
train and test directories in the next step.

```
$ cd $MODEL_WORK_DIR
$ mkdir empty_dir

$ cd annotations
$ echo "{ \"images\": {}, \"categories\": {}}" > empty.json
$ cd $MODEL_WORK_DIR
```

5. Now that you have the raw COCO dataset and annotations files, we need to convert it to the
TF records format in order to use it with the inference script.  We will
do this by running the `create_coco_tf_record.py` file in the TensorFlow
models repo.

Follow [instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md#dependencies) to install the required dependencies (`cocoapi` and `Protobuf 3.0.0`).
Follow the steps below to navigate to the proper directory and point the
script to the raw COCO dataset files that you have downloaded in step 2
and the annotations files that you downloaded and created in step 3.
The `--output_dir` is the location where the TF record files will be
located after the script has completed.

```

# We are going to use an older version of the conversion script to checkout the git commit
$ cd tf_models
$ git checkout 7a9934df2afdf95be9405b4e9f1f2480d748dc40

$ cd research/object_detection/dataset_tools/
$ python create_coco_tf_record.py --logtostderr \
      --train_image_dir="$MODEL_WORK_DIR/empty_dir" \
      --val_image_dir="$MODEL_WORK_DIR/val/val2017" \
      --test_image_dir="$MODEL_WORK_DIR/empty_dir" \
      --train_annotations_file="$MODEL_WORK_DIR/annotations/empty.json" \
      --val_annotations_file="$MODEL_WORK_DIR/annotations/annotations/instances_val2017.json" \
      --testdev_annotations_file="$MODEL_WORK_DIR/annotations/empty.json" \
      --output_dir="$MODEL_WORK_DIR/output"

$ ll $MODEL_WORK_DIR/output
total 1598276
-rw-rw-r--. 1 <user> <group>         0 Nov  2 21:46 coco_testdev.record
-rw-rw-r--. 1 <user> <group>         0 Nov  2 21:46 coco_train.record
-rw-rw-r--. 1 <user> <group> 818336740 Nov  2 21:46 coco_val.record

# Go back to the main models directory and get master code
$ cd $MODEL_WORK_DIR/tf_models
$ git checkout master
```

The `coco_val.record` file is what we will use in this inference example.

6. Download and extract the pre-trained model.
```
$ cd $MODEL_WORK_DIR
$ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/faster_rcnn_resnet50_fp32_coco_pretrained_model.tar.gz
$ tar -xzvf faster_rcnn_resnet50_fp32_coco_pretrained_model.tar.gz
```

7. Clone the [intelai/models](https://github.com/intelai/models) repo.
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

8. Run the `launch_benchmark.py` script from the intelai/models repo
, with the appropriate parameters including: the
`coco_val.record` data location (from step 4), the pre-trained model
`pipeline.config` file and the checkpoint location (from step 5), and the
location of your `tensorflow/models` clone (from step 1).

Run for batch and online inference:
```
$ cd $MODEL_WORK_DIR/models/benchmarks

$ python launch_benchmark.py \
    --data-location $MODEL_WORK_DIR/output/ \
    --model-source-dir $MODEL_WORK_DIR/tf_models \
    --model-name faster_rcnn \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --socket-id 0 \
    --checkpoint $MODEL_WORK_DIR/faster_rcnn_resnet50_fp32_coco \
    --docker-image gcr.io/deeplearning-platform-release/tf-cpu.1-14 \
    -- config_file=pipeline.config
```

Or for accuracy where the `--data-location` is the path the directory
where your `coco_val.record` file is located and the `--in-graph` is
the pre-trained graph located in the pre-trained model directory (from step 5):
```
$ python launch_benchmark.py \
    --model-name faster_rcnn \
    --mode inference \
    --precision fp32 \
    --framework tensorflow \
    --docker-image gcr.io/deeplearning-platform-release/tf-cpu.1-14 \
    --model-source-dir $MODEL_WORK_DIR/tf_models \
    --data-location $MODEL_WORK_DIR/output/ \
    --in-graph $MODEL_WORK_DIR/faster_rcnn_resnet50_fp32_coco/frozen_inference_graph.pb \
    --accuracy-only
```

9. The log file is saved to the value of `--output-dir`.

Below is a sample log file tail when running for batch
and online inference:

```
Time spent : 167.353 seconds.
Time spent per BATCH: 0.167 seconds.
Received these standard args: Namespace(accuracy_only=False, batch_size=1, benchmark_only=False, checkpoint='/checkpoints', data_location='/dataset', framework='tensorflow', input_graph=None, intelai_models='/workspace/intelai_models', mode='inference', model_args=[], model_name='faster_rcnn', model_source_dir='/workspace/models', num_cores=-1, num_inter_threads=2, num_intra_threads=56, precision='fp32', socket_id=0, use_case='object_detection', verbose=True)
Received these custom args: ['--config_file=pipeline.config']
Run model here.
current directory: /workspace/models/research
Running: numactl --cpunodebind=0 --membind=0 python /workspace/intelai_models/inference/fp32/eval.py --num_inter_threads 1 --num_intra_threads 28 --pipeline_config_path /checkpoints/pipeline.config --checkpoint_dir /checkpoints --eval_dir /workspace/models/research/object_detection/log/eval
PYTHONPATH: :/workspace/intelai_models:/workspace/models/research:/workspace/models/research/slim:/workspace/models
RUNCMD: python common/tensorflow/run_tf_benchmark.py --framework=tensorflow --use-case=object_detection --model-name=faster_rcnn --precision=fp32 --mode=inference --model-source-dir=/workspace/models --intelai-models=/workspace/intelai_models --num-cores=-1 --batch-size=1 --data-location=/dataset --socket-id 0 --verbose --checkpoint=/checkpoints         --config_file=pipeline.config
Batch Size: 1
Ran inference with batch size 1
Log location outside container: {--output-dir value}/benchmark_faster_rcnn_inference.log
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

10. To return to where you started from:
```
$ popd
```
