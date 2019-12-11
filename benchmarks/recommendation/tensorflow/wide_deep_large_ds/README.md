# Wide & Deep

This document has instructions for how to run Wide & Deep for the
following modes/precisions:

* [Prepare dataset](#Prepare-dataset)
* [INT8 inference](#int8-inference-instructions)
* [FP32 inference](#fp32-inference-instructions)
* [FP32 Training](#fp32-training-instructions)

## Prepare dataset

1. Download large Kaggle Display Advertising Challenge Dataset from
   http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/

   Note: The dataset does not contain the eval.txt file required for measuring model accuracy. So, download the evaluation
   dataset for accuracy measurement from https://storage.googleapis.com/dataset-uploader/criteo-kaggle/large_version/eval.csv

2. Pre-process the downloaded dataset to tfrecords using [preprocess_csv_tfrecords.py](/models/recommendation/tensorflow/wide_deep_large_ds/dataset/preprocess_csv_tfrecords.py)

    Copy the eval.csv and test.csv into your current working directory (i.e. root of models repo) and launch

    * Launch docker
        ```
        cd /home/<user>/models/

        docker run -it --privileged -u root:root \
                    -w /models \
                    --volume $PWD:/models \
                    docker.io/intelaipg/intel-optimized-tensorflow:nightly-latestprs-bdw \
                   /bin/bash

        ```
    * Process eval dataset
        ```
        python models/recommendation/tensorflow/wide_deep_large_ds/dataset/preprocess_csv_tfrecords.py \
                --inputcsv-datafile eval.csv \
                --calibrationcsv-datafile train.csv 
                --outputfile-name preprocessed_eval
        ```

    * Process test dataset

        The test.txt is in tab-separated values (TSV) format and they must be converted into comma-separated values (CSV) format before doing the pre-processing. On docker console run the below commands to pre-process test datasets
        ```
        $ tr '\t' ',' < test.txt > test.csv
        python models/models/recommendation/tensorflow/wide_deep_large_ds/dataset/preprocess_csv_tfrecords.py \
                --inputcsv-datafile test.csv \
                --outputfile-name preprocessed_test
        ```
        Now preprocessed eval and test datasets will be stored as eval_preprocessed_eval.tfrecords and  test_preprocessed_test.tfrecords in /home/<user>/models/ directory

        Exit out of docker once the dataset pre-processing completes.

## INT8 Inference Instructions

These instructions use the TCMalloc memory allocator, which produces 
better performance results for Int8 precision models with smaller batch sizes. 
If you want to disable the use of TCMalloc, set `--disable-tcmalloc=True` 
when calling `launch_benchmark.py` and the script will run without TCMalloc.

1. Download and extract the pre-trained model.
    ```
    wget https://storage.googleapis.com/intel-optimized-tensorflow/models/wide_deep_int8_pretrained_model.pb
    ```
2. Clone the [intelai/models](https://github.com/intelai/models) repo.

   This repo has the launch script for running the model, which we will
   use in the next step.
   ```
   git clone https://github.com/IntelAI/models.git
   ```
3. Run Accuracy test

    * Running inference to check accuracy, set `--batch-size 1000`
        ```
        cd /home/<user>/models/benchmarks

        python launch_benchmark.py \
            --model-name wide_deep_large_ds \
            --precision int8 \
            --mode inference \
            --framework tensorflow \
            --batch-size 1000 \
            --socket-id 0 \
            --accuracy-only \
            --docker-image docker.io/intelaipg/intel-optimized-tensorflow:nightly-latestprs-bdw \
            --in-graph /root/user/wide_deep_files/wide_deep_int8_pretrained_model.pb \
            --data-location /root/user/wide_deep_files/eval_preprocessed_eval.tfrecords
        ```

4. Run Performance test

   * Running in online inference mode, set `--batch-size 1`

       ``` 
       cd /home/<user>/models/benchmarks

       python launch_benchmark.py \
            --model-name wide_deep_large_ds \
            --precision int8 \
            --mode inference \
            --framework tensorflow \
            --benchmark-only \
            --batch-size 1 \
            --socket-id 0 \
            --docker-image docker.io/intelaipg/intel-optimized-tensorflow:nightly-latestprs-bdw \
            --in-graph $MODEL_WORK_DIR/wide_deep_int8_pretrained_model.pb \
            --data-location $MODEL_WORK_DIR/eval_preprocessed_eval.tfrecords \
            --num-intra-threads 1 --num-inter-threads 1 --num-cores 1 \
            -- num_omp_threads=1
       ```
   * Running in batch inference mode, set `--batch-size 512`
       ``` 
        cd /home/<user>/models/benchmarks
    
        python launch_benchmark.py \
            --model-name wide_deep_large_ds \
            --precision int8 \
            --mode inference \
            --framework tensorflow \
            --benchmark-only \
            --batch-size 512 \
            --socket-id 0 \
            --docker-image docker.io/intelaipg/intel-optimized-tensorflow:nightly-latestprs-bdw \
            --in-graph $MODEL_WORK_DIR/wide_deep_int8_pretrained_model.pb \
            --data-location $MODEL_WORK_DIR/eval_preprocessed_eval.tfrecords \
            --num-intra-threads 28 --num-inter-threads 1 --num-cores 28 \
            -- num_omp_threads=16
       ```
   * The log file is saved to the value of `--output-dir`. The tail of the log output when the script completes 
     should look something like this:
        ```
        --------------------------------------------------
        Total test records           :  2000000
        Batch size is                :  512
        Number of batches            :  3907
        Classification accuracy (%)  :  77.6405
        No of correct predictions    :  1552720
        Inference duration (seconds) :  4.2531
        Avergare Latency (ms/batch)  :  1.1173
        Throughput is (records/sec)  :  696784.187
        --------------------------------------------------
        Ran inference with batch size 512
        Log location outside container:  {--output-dir value}/benchmark_wide_deep_large_ds_inference_int8_20190225_061815.log
        ```

## FP32 Inference Instructions

1. Download and extract the pre-trained model.
    ```
    wget https://storage.googleapis.com/intel-optimized-tensorflow/models/wide_deep_fp32_pretrained_model.pb
    ```
2. Clone the [intelai/models](https://github.com/intelai/models) repo.

   This repo has the launch script for running the model, which we will
   use in the next step.

    ```
    git clone https://github.com/IntelAI/models.git
    ```
3. Run Accuracy test

    * Running inference for checking accuracy, set `--batch-size 1000`
        ```
        cd /home/<user>/models/benchmarks

        python launch_benchmark.py
            --model-name wide_deep_large_ds \
            --precision fp32 \
            --mode inference \
            --framework tensorflow \
            --batch-size 1000 \
            --socket-id 0 \
            --accuracy-only \
            --docker-image docker.io/intelaipg/intel-optimized-tensorflow:latest \
            --in-graph $MODEL_WORK_DIR/wide_deep_fp32_pretrained_model.pb \
            --data-location $MODEL_WORK_DIR/eval_preprocessed_eval.tfrecords
       ```

4. Run Performance test

    * Running in online inference mode, set `--batch-size 1`

        ```
        cd /home/<user>/models/benchmarks

        python launch_benchmark.py
            --model-name wide_deep_large_ds \
            --precision fp32 \
            --mode inference \
            --framework tensorflow \
            --benchmark-only \
            --batch-size 1 \
            --socket-id 0 \
            --docker-image gcr.io/deeplearning-platform-release/tf-cpu.1-14 \
            --in-graph $MODEL_WORK_DIR/wide_deep_fp32_pretrained_model.pb \
            --data-location $MODEL_WORK_DIR/eval_preprocessed_eval.tfrecord \
            --num-intra-threads 1 --num-inter-threads 1 --num-cores 1 \
            -- num_omp_threads=1
        ```
    * Running in batch inference mode, set `--batch-size 512`
        ```
        cd /home/<user>/models/benchmarks

        python launch_benchmark.py
            --model-name wide_deep_large_ds \
            --precision fp32 \
            --mode inference \
            --framework tensorflow \
            --benchmark-only \
            --batch-size 512 \
            --socket-id 0 \
            --docker-image gcr.io/deeplearning-platform-release/tf-cpu.1-14 \
            --in-graph $MODEL_WORK_DIR/wide_deep_fp32_pretrained_model.pb \
            --data-location $MODEL_WORK_DIR/eval_preprocessed_eval.tfrecords \
            --num-intra-threads 28 --num-inter-threads 1 --num-cores 28 \
            -- num_omp_threads=20
        ```
    * The log file is saved to the value of `--output-dir`. The tail of the log output when the script completes 
        should look something like this:
        ```
        --------------------------------------------------
        Total test records           :  2000000
        Batch size is                :  512
        Number of batches            :  3907
        Classification accuracy (%)  :  77.6693
        No of correct predictions    :  1553386
        Inference duration (seconds) :  5.6724
        Avergare Latency (ms/batch)  :  1.4902
        Throughput is (records/sec)  :  343560.261
        --------------------------------------------------
        Ran inference with batch size 512
        Log location outside container: {--output-dir value}/benchmark_wide_deep_large_ds_inference_fp32_20190225_062206.log
        ```

4. To return to where you started from:
```
$ popd
```
## FP32 Training Instructions

1. Download large Kaggle Display Advertising Challenge Dataset from
   http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/

   Download the large version of train dataset from https://storage.googleapis.com/dataset-uploader/criteo-kaggle/large_version/train.csv
   
   Download the large version of evaluation dataset from https://storage.googleapis.com/dataset-uploader/criteo-kaggle/large_version/eval.csv

## Train Wide and Deep Model   

2. Train the model by providing location of train.csv, eval.csv 

    * Train the model (The model will be trained for 10 epochs if -- steps is not specified)
        ```
        $ python launch_benchmark.py --model-name wide_deep_large_ds \
           --precision fp32 \
           --mode training  \
           --framework tensorflow \
           --batch-size 512 \
           --data-location /root/dataset \
           --docker-image gcr.io/deeplearning-platform-release/tf-cpu.1-14
        
        ```
