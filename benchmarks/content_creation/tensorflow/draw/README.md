# DRAW

This document has instructions for how to run DRAW for the following
modes/precisions:
* [FP32 inference](#fp32-inference-instructions)

## FP32 Inference Instructions

1. Download the [MNIST dataset](http://yann.lecun.com/exdb/mnist/):

   ```
   $ mkdir mnist
   $ cd mnist
   $ wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
   $ wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
   $ wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
   $ wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
   ```

   The mnist directory will be passed as the dataset location when we
   run the benchmarking script in step 4.

2. A link to download the pre-trained model is coming soon.

3. Clone this [intelai/models](https://github.com/IntelAI/models) repo,
   which contains the scripts that we will be using to run benchmarking
   for DRAW.  After the clone has completed, navigate to the `benchmarks`
   directory in the repository.

   ```
   $ git clone https://github.com/IntelAI/models.git
   $ cd models/benchmarks
   ```

4. Run benchmarking for either throughput or latency using the commands
   below. Replace in the path to the `--data-location` with your `mnist`
   dataset directory from step 1 and the `--checkpoint` files that you
   downloaded and extracted in step 2.

   * Run benchmarking for latency (with `--batch-size 1`):
     ```
        python launch_benchmark.py \
	        --precision fp32 \
            --model-name draw \
            --mode inference \
            --framework tensorflow \
            --docker-image intelaipg/intel-optimized-tensorflow:latest-devel-mkl-py3 \
            --checkpoint /home/myuser/draw_fp32_pretrained_model \
            --data-location /home/myuser/mnist \
            --batch-size 1 \
            --socket-id 0
     ```
    * Run benchmarking for throughput (with `--batch-size 100`):
      ```
        python launch_benchmark.py \
	        --precision fp32 \
            --model-name draw \
            --mode inference \
            --framework tensorflow \
            --docker-image intelaipg/intel-optimized-tensorflow:latest-devel-mkl-py3 \
            --checkpoint /home/myuser/draw_fp32_pretrained_model \
            --data-location /home/myuser/mnist \
            --batch-size 100 \
            --socket-id 0
      ```
      Note that the `--verbose` flag can be added to any of the above
      commands to get additional debug output.

4. The log files for each benchmarking run are saved at:
   `intelai/models/benchmarks/common/tensorflow/logs`.

   * Below is a sample log file tail when benchmarking latency:
     ```
     ...
     Elapsed Time 0.006622
     Elapsed Time 0.006636
     Elapsed Time 0.006602
     Batchsize: 1
     Time spent per BATCH: 6.6667 ms
     Total samples/sec: 149.9996 samples/s
     Outputs saved in file: /home/myuser/mnist/draw_data.npy
     lscpu_path_cmd = command -v lscpu
     lscpu located here: b'/usr/bin/lscpu'
     Ran inference with batch size 1
     Log location outside container: /home/myuser/intelai/models/benchmarks/common/tensorflow/logs/benchmark_draw_inference_fp32_20190123_012947.log
     ```

   * Below is a sample log file tail when benchmarking throughput:
     ```
     Elapsed Time 0.028355
     Elapsed Time 0.028221
     Elapsed Time 0.028183
     Batchsize: 100
     Time spent per BATCH: 28.1952 ms
     Total samples/sec: 3546.7006 samples/s
     Outputs saved in file: /home/myuser/mnist/draw_data.npy
     lscpu_path_cmd = command -v lscpu
     lscpu located here: b'/usr/bin/lscpu'
     Ran inference with batch size 100
     Log location outside container: /home/myuser/intelai/models/benchmarks/common/tensorflow/logs/benchmark_draw_inference_fp32_20190123_013432.log
     ```