# Transformer LT Official

This document has instructions for how to run Transformer LT Official for the
following modes/precisions:
* [FP32 inference](#fp32-inference-instructions)

## FP32 Inference Instructions
1. Download the pre-trained model and test data.

    Download and extract the packaged pre-trained model and dataset `transformer_lt_official_fp32_pretrained_model.tar.gz`

    ```
    $ cd ~
    $ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/transformer_lt_official_fp32_pretrained_model.tar.gz
    $ tar -xzvf transformer_lt_official_fp32_pretrained_model.tar.gz
    $ export TLT_DATA_LOCATION=$(pwd)/transformer_lt_official_fp32_pretrained_model/data
    $ export TLT_MODEL=$(pwd)/transformer_lt_official_fp32_pretrained_model/graph/fp32_graphdef.pb

    ```
3. Clone this [intelai/models](https://github.com/IntelAI/models)
repository:
    ```
    $ git clone https://github.com/IntelAI/models.git
    $ cd models/benchmarks
    ```
4. For specifying a docker image that the model server should run with, you should use the --docker-image arg. This will pull the TensorFlow Serving image and run the model in docker. The client benchmarking script will then be launched from a virtualenv on bare metal and make requests to the serving container over GRPC.

5. Transformer-LT can be run for measuring online inference or batch inference using following command.
    * For online inference use `--batch-size=1` 
        ```
        $ python launch_benchmark.py \
            --in-graph $TLT_MODEL \
            --model-name=transformer_lt_official \
            --framework=tensorflow_serving \
            --precision=fp32 \
            --mode inference \
            --batch-size=1 \
            --docker-image=intel/intel-optimized-tensorflow-serving:2.3.0 \
            --data-location $TLT_DATA_LOCATION \
            --benchmark-only
        ```
        Example log tail when running for online inference:
        ```
        Iteration 15: 1.181 sec
        Iteration 16: 1.125 sec
        Iteration 17: 1.210 sec
        Iteration 18: 1.131 sec
        Iteration 19: 1.015 sec
        Iteration 20: 1.086 sec
        Inferencing time: 11.211763143539429
        Batch size = 1
        Latency: 1121.176 ms
        Throughput: 0.892 sentences/sec
        + docker rm -f tfserving_19410
        tfserving_19410
        + popd
        /home/<user>/models/benchmarks
        + rm -rf workspace
        + echo 'Log output location: /home/<user>/models/benchmarks/common/tensorflow_serving/logs/benchmark_transformer_lt_official_inference_fp32_20200709_164214.log'
        + tee -a /home/<user>/models/benchmarks/common/tensorflow_serving/logs/benchmark_transformer_lt_official_inference_fp32_20200709_164214.log
        Log output location: /home/<user>/models/benchmarks/common/tensorflow_serving/logs/benchmark_transformer_lt_official_inference_fp32_20200709_164214.log
        ```

    * For online inference use `--batch-size=128`
        ```
        $ python launch_benchmark.py \
            --in-graph $TLT_MODEL \
            --model-name=transformer_lt_official \
            --framework=tensorflow_serving \
            --precision=fp32 \
            --mode inference \
            --batch-size=128 \
            --docker-image=intel/intel-optimized-tensorflow-serving:2.3.0 \
            --data-location $TLT_DATA_LOCATION \
            --benchmark-only
        ```
        Example log tail when running for online inference:
        ```
        Iteration 15: 6.117 sec
        Iteration 16: 5.773 sec
        Iteration 17: 5.309 sec
        Iteration 18: 4.300 sec
        Iteration 19: 5.214 sec
        Iteration 20: 4.216 sec
        Inferencing time: 56.74819374084473
        Batch size = 128
        Throughput: 22.556 sentences/sec
        + docker rm -f tfserving_7791
        tfserving_7791
        + popd
        /home/<user>/models/benchmarks
        + rm -rf workspace
        + echo 'Log output location: /home/<user>/models/benchmarks/common/tensorflow_serving/logs/benchmark_transformer_lt_official_inference_fp32_20200709_164621.log'
        + tee -a /home/<user>/models/benchmarks/common/tensorflow_serving/logs/benchmark_transformer_lt_official_inference_fp32_20200709_164621.log
        Log output location: /home/<user>/models/benchmarks/common/tensorflow_serving/logs/benchmark_transformer_lt_official_inference_fp32_20200709_164621.log
        ```
