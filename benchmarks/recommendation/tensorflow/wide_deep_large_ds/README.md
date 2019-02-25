# Wide & Deep

This document has instructions for how to run Wide & Deep benchmark for the
following modes/precisions:

* [INT8 inference](#int8-inference-instructions)
* [FP32 inference](#fp32-inference-instructions)

Benchmarking instructions and scripts for model training coming later.

## INT8 Inference Instructions

 
1. Download large <> dataset income dataset from <>: 
   
   To be updated post dataset approval
       
2. Pre-process the downloaded dataset to tfrecords using [preprocess_csv_tfrecords.py](/models/recommendation/tensorflow/wide_deep_large_ds/dataset/preprocess_csv_tfrecords.py)
   
    ```
	$ python3.6 preprocess_csv_tfrecords.py --csv-datafile eval.csv 
    ```
   
3. Clone the [intelai/models](https://github.com/intelai/models) repo.

   This repo has the launch script for running benchmarks, which we will
   use in the next step.

    ```
    $ git clone https://github.com/IntelAI/models.git
    ```
4. How to run benchmarks

   * Running benchmarks in latency mode, set `--batch-size 1`
       ``` 
       $ cd /home/myuser/models/benchmarks

       $ python launch_benchmark.py 
            --model-name wide_deep_large_ds \
            --precision int8 \
            --mode inference \
            --framework tensorflow \
            --benchmark-only \
            --batch-size 1 \
            --socket-id 0 \
            --docker-image tensorflow/tensorflow:latest-mkl \
            --in-graph /root/user/wide_deep_files/int8_wide_deep_final.pb \
            --data-location /root/user/wide_deep_files/preprocessed_eval.tfrecords 
       ```
   * Running benchmarks in throughput mode, set `--batch-size 1024`
       ``` 
       $ cd /home/myuser/models/benchmarks
    
        $ python launch_benchmark.py 
            --model-name wide_deep_large_ds \
            --precision int8 \
            --mode inference \
            --framework tensorflow \
            --benchmark-only \
            --batch-size 1024 \
            --socket-id 0 \
            --docker-image tensorflow/tensorflow:latest-mkl \
            --in-graph /root/user/wide_deep_files/int8_wide_deep_final.pb \
            --data-location /root/user/wide_deep_files/preprocessed_eval.tfrecords 
       ```
6. The log file is saved to the value of `--output-dir`.

   The tail of the log output when the benchmarking completes should look
   something like this:

    ```
    
    --------------------------------------------------
    Total test records           :  2000000
    No of correct predicitons    :  1549508
    Batch size is                :  1024
    Number of batches            :  1954
    Classification accuracy (%)  :  77.4754
    Inference duration (seconds) :  1.9765
    Latency (millisecond/batch)  :  0.000988
    Throughput is (records/sec)  :  1151892.25
    --------------------------------------------------
    numactl --cpunodebind=0 --membind=0 python /workspace/intelai_models/int8/inference.py --input-graph=/in_graph/int8_wide_deep_final.pb --inter-op-parallelism-threads=28 --intra-op-parallelism-threads=1 --omp-num-threads=1 --batch-size=1024 --kmp-blocktime=0 --datafile-path=/dataset
    Ran inference with batch size 1024
    Log location outside container:  {--output-dir value}/benchmark_wide_deep_large_ds_inference_int8_20190225_061815.log
    ```

## FP32 Inference Instructions

1. Download large <> dataset income dataset from <>: 
   
   To be updated post dataset approval
       
2. Pre-process the downloaded dataset to tfrecords using [preprocess_csv_tfrecords.py](../../../../models/recommendation/tensorflow/wide_deep_large_ds/dataset/preprocess_csv_tfrecords.py)
   
   ```
    $ python3.6 preprocess_csv_tfrecords.py --csv-datafile eval.csv 
   ```
3. Clone the [intelai/models](https://github.com/intelai/models) repo.

   This repo has the launch script for running benchmarks, which we will
   use in the next step.

    ```
    $ git clone https://github.com/IntelAI/models.git
    ```
4. How to run benchmarks

   * Running benchmarks in latency mode, set `--batch-size 1`
       ``` 
       $ cd /home/myuser/models/benchmarks

       $ python launch_benchmark.py 
            --model-name wide_deep_large_ds \
            --precision fp32 \
            --mode inference \
            --framework tensorflow \
            --benchmark-only \
            --batch-size 1 \
            --socket-id 0 \
            --docker-image tensorflow/tensorflow:latest-mkl \
            --in-graph /root/user/wide_deep_files/fp32_wide_deep_final.pb \
            --data-location /root/user/wide_deep_files/preprocessed_eval.tfrecords 
       ```
   * Running benchmarks in throughput mode, set `--batch-size 1024`
       ``` 
       $ cd /home/myuser/models/benchmarks
    
        $ python launch_benchmark.py 
            --model-name wide_deep_large_ds \
            --precision fp32 \
            --mode inference \
            --framework tensorflow \
            --benchmark-only \
            --batch-size 1024 \
            --socket-id 0 \
            --docker-image tensorflow/tensorflow:latest-mkl \
            --in-graph /root/user/wide_deep_files/fp32_wide_deep_final.pb \
            --data-location /root/user/wide_deep_files/preprocessed_eval.tfrecords 
       ```
6. The log file is saved to the value of `--output-dir`.

   The tail of the log output when the benchmarking completes should look
   something like this:

    ```

    --------------------------------------------------
    Total test records           :  2000000
    No of correct predicitons    :  1550447
    Batch size is                :  1024
    Number of batches            :  1954
    Classification accuracy (%)  :  77.5223
    Inference duration (seconds) :  3.4977
    Latency (millisecond/batch)  :  0.001749
    Throughput is (records/sec)  :  571802.228
    --------------------------------------------------
    numactl --cpunodebind=0 --membind=0 python /workspace/intelai_models/int8/inference.py --input-graph=/in_graph/fp32_wide_deep_final.pb --inter-op-parallelism-threads=28 --intra-op-parallelism-threads=1 --omp-num-threads=1 --batch-size=1024 --kmp-blocktime=0 --datafile-path=/dataset
    Ran inference with batch size 1024
    Log location outside container: {--output-dir value}/benchmark_wide_deep_large_ds_inference_fp32_20190225_062206.log
    
    ```
