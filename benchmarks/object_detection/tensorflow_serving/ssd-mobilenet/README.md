# SSD-Mobilenet

This document has instructions for how to run ssd-mobilenet for the
following modes/precisions:
* [FP32 inference](#fp32-inference-instructions)

## FP32 Inference Instructions
1. Download the 2017 validation COCO dataset (~780MB) (**note**: do not convert the COCO dataset to TF records format):
   
   ```
   cd ~
   mkdir -p coco/val
   wget http://images.cocodataset.org/zips/val2017.zip
   unzip val2017.zip -d coco/val
   export COCO_VAL_DATA=$(pwd)/coco/val/val2017
   ```
2. Download the pre-trained model.
    ```
    cd ~
    wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz
    tar -xzvf ssd_mobilenet_v1_coco_2018_01_28.tar.gz
    export SAVED_MODEL=$(pwd)/ssd_mobilenet_v1_coco_2018_01_28/saved_model/saved_model.pb
    ```
3. Clone this [intelai/models](https://github.com/IntelAI/models)
repository:
    ```
    git clone https://github.com/IntelAI/models.git
    cd models/benchmarks
    ```
4. For specifying a docker image that the model server should run with, you should use the --docker-image arg. This will pull the TensorFlow Serving image and run the model in docker. The client benchmarking script will then be launched from a virtualenv on bare metal and make requests to the serving container over GRPC.

5. SSD-Mobilenet can be run for measuring online inference or batch inference using following command.
    * For online inference use `--batch-size=1` 
        ```
        python launch_benchmark.py \
            --in-graph $SAVED_MODEL \
            --model-name=ssd-mobilenet \
            --framework=tensorflow_serving \
            --precision=fp32 \
            --mode inference \
            --batch-size=1 \
            --docker-image=intel/intel-optimized-tensorflow-serving:latest \
            --data-location $COCO_VAL_DATA \
            --benchmark-only
        ```
        Example log tail when running for online inference:
        ```
        Iteration 15: 0.030 sec
        Iteration 16: 0.029 sec
        Iteration 17: 0.030 sec
        Iteration 18: 0.031 sec
        Iteration 19: 0.028 sec
        Iteration 20: 0.030 sec
        Average time: 0.030 sec
        Batch size = 1
        Latency: 29.632 ms
        Throughput: 33.747 images/sec
        + docker rm -f tfserving_4697
        tfserving_4697
        + popd
        /home/<user>/models/benchmarks
        + rm -rf workspace
        + echo 'Log output location: /home/<user>/models/benchmarks/common/tensorflow_serving/logs/benchmark_ssd-mobilenet_inference_fp32_20200709_105820.log'
        + tee -a /home/<user>/models/benchmarks/common/tensorflow_serving/logs/benchmark_ssd-mobilenet_inference_fp32_20200709_105820.log
        Log output location: /home/<user>/models/benchmarks/common/tensorflow_serving/logs/benchmark_ssd-mobilenet_inference_fp32_20200709_105820.log
        ```

    * For online inference use `--batch-size=128`
        ```
        python launch_benchmark.py \
            --in-graph $SAVED_MODEL \
            --model-name=ssd-mobilenet \
            --framework=tensorflow_serving \
            --precision=fp32 \
            --mode inference \
            --batch-size=128 \
            --docker-image=intel/intel-optimized-tensorflow-serving:latest \
            --data-location $COCO_VAL_DATA \
            --benchmark-only
        ```
        Example log tail when running for online inference:
        ```
        Iteration 10: 2.256 sec
        Iteration 11: 2.243 sec
        Iteration 12: 2.266 sec
        Iteration 13: 2.070 sec
        Iteration 14: 2.070 sec
        Iteration 15: 2.204 sec
        Iteration 16: 2.189 sec
        Iteration 17: 2.292 sec
        Iteration 18: 2.113 sec
        Iteration 19: 2.260 sec
        Iteration 20: 2.246 sec
        Average time: 2.195 sec
        Batch size = 128
        Throughput: 58.307 images/sec
        + docker rm -f tfserving_21180
        tfserving_21180
        + popd
        /home/<user>/models/benchmarks
        + rm -rf workspace
        + echo 'Log output location: /home/<user>/models/benchmarks/common/tensorflow_serving/logs/benchmark_ssd-mobilenet_inference_fp32_20200709_110609.log'
        + tee -a /home/<user>/models/benchmarks/common/tensorflow_serving/logs/benchmark_ssd-mobilenet_inference_fp32_20200709_110609.log
        Log output location: /home/<user>/models/benchmarks/common/tensorflow_serving/logs/benchmark_ssd-mobilenet_inference_fp32_20200709_110609.log
        ```
