# ResNet101

This document has instructions for how to run ResNet101 for the
following platform:
* [FP32 inference](#fp32-inference-instructions)


## FP32 Inference Instructions

1. Clone the 
[intelai/models](https://github.com/intelai/models)
repository
    ```
    $ git clone git@github.com:IntelAI/models.git
    ```

2. Download the pre-trained ResNet50 model:

    ```
    $ wget https://storage.cloud.google.com/intel-optimized-tensorflow/models/resenet101_fp32_pretrained_model.pb
    ```
3. Download ImageNet dataset.

    This step is required only required for running accuracy, for running benchmark we do not need to provide dataset.
    
    Register and download the ImageNet dataset. Once you have the raw ImageNet dataset downloaded, we need to convert 
    it to the TFRecord format. The TensorFlow models repo provides
    [scripts and instructions](https://github.com/tensorflow/models/tree/master/research/slim#an-automated-script-for-processing-imagenet-data)
    to download, process and convert the ImageNet dataset to the TF records format. After converting data, you should have a directory 
    with the sharded dataset something like below, we only need `validation-*` files, discard `train-*` files:
    ```
    $ ll /home/myuser/datasets/ImageNet_TFRecords
    -rw-r--r--. 1 user 143009929 Jun 20 14:53 train-00000-of-01024
    -rw-r--r--. 1 user 144699468 Jun 20 14:53 train-00001-of-01024
    -rw-r--r--. 1 user 138428833 Jun 20 14:53 train-00002-of-01024
    ...
    -rw-r--r--. 1 user 143137777 Jun 20 15:08 train-01022-of-01024
    -rw-r--r--. 1 user 143315487 Jun 20 15:08 train-01023-of-01024
    -rw-r--r--. 1 user  52223858 Jun 20 15:08 validation-00000-of-00128
    -rw-r--r--. 1 user  51019711 Jun 20 15:08 validation-00001-of-00128
    -rw-r--r--. 1 user  51520046 Jun 20 15:08 validation-00002-of-00128
    ...
    -rw-r--r--. 1 user  52508270 Jun 20 15:09 validation-00126-of-00128
    -rw-r--r--. 1 user  55292089 Jun 20 15:09 validation-00127-of-00128
    ```
4. Run the benchmark. 
    
    For latency measurements set `--batch-size 1` and for throughput benchmarking set `--batch-size 128` 

    ```
    $ cd /home/myuser/models/benchmarks
    $ python launch_benchmark.py 
        --framework tensorflow 
        --platform fp32 
        --mode inference 
        --model-name resnet101 
        --batch-size 128  
        --docker-image intelaipg/intel-optimized-tensorflow:latest-devel-mkl 
        --in-graph /home/myuser/trained_models/resenet101_fp32_pretrained_model.pb  
        --single-socket 
        --verbose
    ```

    The log file is saved to: `models/benchmarks/common/tensorflow/logs/`.
    
    The tail of the log output when the benchmarking completes should look something like this:

      ```
        steps = 10, 1342.80359717 images/sec
        steps = 20, 670.767434629 images/sec
        steps = 30, 446.319515464 images/sec
        steps = 40, 334.314206698 images/sec
        steps = 50, 267.251707323 images/sec
        steps = 60, 222.571395923 images/sec
        steps = 70, 190.724044039 images/sec
        steps = 80, 166.881224428 images/sec
        steps = 90, 148.365949039 images/sec
        steps = 100, 133.594262281 images/sec
        lscpu_path_cmd = command -v lscpu
        lscpu located here: /usr/bin/lscpu
        Received these standard args: Namespace(accuracy_only=False, batch_size=128, benchmark_only=False, checkpoint=None, data_location='/dataset', framework='tensorflow', input_graph='/in_graph/resenet101_fp32_model.pb', intelai_models='/workspace/intelai_models', mode='inference', model_args=[], model_name='resnet101', model_source_dir='/workspace/models', num_cores=-1, num_inter_threads=2, num_intra_threads=56, platform='fp32', single_socket=True, socket_id=0, use_case='image_recognition', verbose=True)
        Received these custom args: []
        Current directory: /workspace/benchmarks
        Running: numactl --cpunodebind=0 --membind=0 python /workspace/intelai_models/fp32/benchmark.py --batch_size=128 --num_inter_threads=2 --input_graph=/in_graph/resenet101_fp32_model.pb --num_intra_threads=56
        PYTHONPATH: :/workspace/intelai_models:/workspace/benchmarks/common/tensorflow:/workspace/benchmarks
        RUNCMD: python common/tensorflow/run_tf_benchmark.py --framework=tensorflow --use-case=image_recognition --model-name=resnet101 --platform=fp32 --mode=inference --model-source-dir=/workspace/models --intelai-models=/workspace/intelai_models --num-cores=-1 --batch-size=128 --single-socket --verbose --in-graph=/in_graph/resenet101_fp32_model.pb       --data-location=/dataset
        Batch Size: 128
        Ran inference with batch size 128
        Log location outside container: /home/myuser/resnet101/intel-models/benchmarks/common/tensorflow/logs/benchmark_resnet101_inference_fp32_20181205_194744.log
      ```

5. Run for accuracy
    ```
    $ cd /home/myuser/models/benchmarks
    $ python launch_benchmark.py 
        --framework tensorflow 
        --platform fp32 
        --mode inference 
        --model-name resnet101 
        --batch-size 100  
        --docker-image intelaipg/intel-optimized-tensorflow:latest-devel-mkl 
        --in-graph /home/myuser/trained_models/resenet101_fp32_pretrained_model.pb
        --data-location /home/myuser/imagenet_validation_dataset 
        --accuracy-only  
        --single-socket 
        --verbose
    ```

    The log file is saved to: `/home/myuser/resnet101/intel-models/benchmarks/common/tensorflow/logs/`.
    
    The tail of the log output when the benchmarking completes should look something like this:

      ```
        Processed 49300 images. (Top1 accuracy, Top5 accuracy) = (0.7641, 0.9290)
        Processed 49400 images. (Top1 accuracy, Top5 accuracy) = (0.7639, 0.9289)
        Processed 49500 images. (Top1 accuracy, Top5 accuracy) = (0.7639, 0.9289)
        Processed 49600 images. (Top1 accuracy, Top5 accuracy) = (0.7638, 0.9289)
        Processed 49700 images. (Top1 accuracy, Top5 accuracy) = (0.7638, 0.9288)
        Processed 49800 images. (Top1 accuracy, Top5 accuracy) = (0.7640, 0.9289)
        Processed 49900 images. (Top1 accuracy, Top5 accuracy) = (0.7640, 0.9289)
        Processed 50000 images. (Top1 accuracy, Top5 accuracy) = (0.7640, 0.9289)
        lscpu_path_cmd = command -v lscpu
        lscpu located here: /usr/bin/lscpu
        Received these standard args: Namespace(accuracy_only=True, batch_size=100, benchmark_only=False, checkpoint=None, data_location='/dataset', framework='tensorflow', input_graph='/in_graph/resenet101_fp32_model.pb', intelai_models='/workspace/intelai_models', mode='inference', model_args=[], model_name='resnet101', model_source_dir='/workspace/models', num_cores=-1, num_inter_threads=2, num_intra_threads=56, platform='fp32', single_socket=True, socket_id=0, use_case='image_recognition', verbose=True)
        Received these custom args: []
        Current directory: /workspace/benchmarks
        Running: numactl --cpunodebind=0 --membind=0 python /workspace/intelai_models/fp32/accuracy.py --batch_size=100 --num_inter_threads=2 --input_graph=/in_graph/resenet101_fp32_model.pb --num_intra_threads=56 --data_location=/dataset
        PYTHONPATH: :/workspace/intelai_models:/workspace/benchmarks/common/tensorflow:/workspace/benchmarks
        RUNCMD: python common/tensorflow/run_tf_benchmark.py --framework=tensorflow --use-case=image_recognition --model-name=resnet101 --platform=fp32 --mode=inference --model-source-dir=/workspace/models --intelai-models=/workspace/intelai_models --num-cores=-1 --batch-size=100 --single-socket --accuracy-only  --verbose --in-graph=/in_graph/resenet101_fp32_model.pb --data-location=/dataset
        Batch Size: 100
        Ran inference with batch size 100
        Log location outside container: /home/myuser/resnet101/intel-models/benchmarks/common/tensorflow/logs/benchmark_resnet101_inference_fp32_20181207_221503.log
    ```
