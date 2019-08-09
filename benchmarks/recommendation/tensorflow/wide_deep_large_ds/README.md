# Wide & Deep

This document has instructions for how to run Wide & Deep for the
following modes/precisions:

* [Prepare dataset](#Prepare-dataset)
* [INT8 inference](#int8-inference-instructions)
* [FP32 inference](#fp32-inference-instructions)

Instructions and scripts for model training coming later.

## Prepare dataset

1. Store the path to the current directory and clone the [intelai/models](https://github.com/intelai/models) repo.
    ```
    $ mkdir wide_deep_large_ds
    $ cd wide_deep_large_ds
    $ MODEL_WORK_DIR=${MODEL_WORK_DIR:=`pwd`}
    $ pushd $MODEL_WORK_DIR

    $ git clone https://github.com/IntelAI/models.git
    ```

2. Download large Kaggle Display Advertising Challenge Dataset from
   http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/

   Note: The dataset does not contain the eval.txt file required for measuring model accuracy. So, download the evaluation
   dataset for accuracy measurement from https://storage.googleapis.com/dataset-uploader/criteo-kaggle/large_version/eval.csv

3. Pre-process the downloaded dataset to tfrecords using [preprocess_csv_tfrecords.py](/models/recommendation/tensorflow/wide_deep_large_ds/dataset/preprocess_csv_tfrecords.py)

    Copy the eval.csv and test.csv into your current working directory (i.e. root of models repo) and launch

    * Launch docker
        ```
        $ cd $MODEL_WORK_DIR/models/

        $ docker run -it --privileged -u root:root \
                    -w /models \
                    --volume $MODEL_WORK_DIR:/models \
                    docker.io/intelaipg/intel-optimized-tensorflow:nightly-latestprs-bdw \
                   /bin/bash

        ```
    * Process eval dataset
        ```
        $ python models/models/recommendation/tensorflow/wide_deep_large_ds/dataset/preprocess_csv_tfrecords.py \
                --csv-datafile eval.csv \
                --outputfile-name preprocessed_eval
        ```

    * Process test dataset

        The test.txt is in tab-separated values (TSV) format and they must be converted into comma-separated values (CSV) format before doing the pre-processing. On docker console run the below commands to pre-process test datasets
        ```
        $ tr '\t' ',' < test.txt > test.csv

        $ python models/models/recommendation/tensorflow/wide_deep_large_ds/dataset/preprocess_csv_tfrecords.py \
                --csv-datafile test.csv \
                --outputfile-name preprocessed_test
        ```
        Now preprocessed eval and test datasets will be stored as eval_preprocessed_eval.tfrecords and  test_preprocessed_test.tfrecords in $MODEL_WORK_DIR/models/ directory

4. Exit out of docker once the dataset pre-processing completes.
    ```
    $ exit
    ```

## INT8 Inference Instructions

These instructions use the TCMalloc memory allocator, which produces 
better performance results for Int8 precision models with smaller batch sizes. 
If you want to disable the use of TCMalloc, set `--disable-tcmalloc=True` 
when calling `launch_benchmark.py` and the script will run without TCMalloc.

1. Download and extract the pre-trained model.
    ```
    $ cd $MODEL_WORK_DIR

    $ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/wide_deep_int8_pretrained_model.pb
    ```

2. Run Accuracy test

    * Running inference to check accuracy, set `--batch-size 1000`
        ```
        $ cd $MODEL_WORK_DIR/models/benchmarks

        $ python launch_benchmark.py \
            --model-name wide_deep_large_ds \
            --precision int8 \
            --mode inference \
            --framework tensorflow \
            --batch-size 1000 \
            --socket-id 0 \
            --accuracy-only \
            --docker-image docker.io/intelaipg/intel-optimized-tensorflow:nightly-latestprs-bdw \
            --in-graph $MODEL_WORK_DIR/wide_deep_int8_pretrained_model.pb \
            --data-location $MODEL_WORK_DIR/eval_preprocessed_eval.tfrecords
        ```

3. Run Performance test

   * Running in online inference mode, set `--batch-size 1`

       ``` 
       $ cd $MODEL_WORK_DIR/models/benchmarks

       $ python launch_benchmark.py \
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
            -- num_parallel_batches=1
       ```
   * Running in batch inference mode, set `--batch-size 512`
       ``` 
        $ cd $MODEL_WORK_DIR/models/benchmarks
    
        $ python launch_benchmark.py \
            --model-name wide_deep_large_ds \
            --precision int8 \
            --mode inference \
            --framework tensorflow \
            --benchmark-only \
            --batch-size 512 \
            --socket-id 0 \
            --docker-image docker.io/intelaipg/intel-optimized-tensorflow:nightly-latestprs-bdw \
            --in-graph $MODEL_WORK_DIR/wide_deep_int8_pretrained_model.pb \
            --data-location $MODEL_WORK_DIR/eval_preprocessed_eval.tfrecords
       ```
   * The log file is saved to the value of `--output-dir`. The tail of the log output when the script completes 
     should look something like this:
        ```
        --------------------------------------------------
        Total test records           :  2000000
        No of correct predictions    :  1549508
        Batch size is                :  512
        Number of batches            :  1954
        Classification accuracy (%)  :  77.5087
        Inference duration (seconds) :  1.9765
        Latency (millisecond/batch)  :  0.000988
        Throughput is (records/sec)  :  1151892.25
        --------------------------------------------------
        Ran inference with batch size 512
        Log location outside container:  {--output-dir value}/benchmark_wide_deep_large_ds_inference_int8_20190225_061815.log
        ```

4. To return to where you started from:
```
$ popd
```

## FP32 Inference Instructions

1. Download and extract the pre-trained model.
    ```
    $ cd $MODEL_WORK_DIR

    $ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/wide_deep_fp32_pretrained_model.pb
    ```

2. Run Accuracy test

    * Running inference for checking accuracy, set `--batch-size 1000`
        ```
        $ cd $MODEL_WORK_DIR/models/benchmarks

        $ python launch_benchmark.py \
            --model-name wide_deep_large_ds \
            --precision fp32 \
            --mode inference \
            --framework tensorflow \
            --batch-size 1000 \
            --socket-id 0 \
            --accuracy-only \
            --docker-image gcr.io/deeplearning-platform-release/tf-cpu.1-14 \
            --in-graph $MODEL_WORK_DIR/wide_deep_fp32_pretrained_model.pb \
            --data-location $MODEL_WORK_DIR/eval_preprocessed_eval.tfrecords
       ```

3. Run Performance test

    * Running in online inference mode, set `--batch-size 1`

        ```
        $ cd $MODEL_WORK_DIR/models/benchmarks

        $ python launch_benchmark.py \
            --model-name wide_deep_large_ds \
            --precision fp32 \
            --mode inference \
            --framework tensorflow \
            --benchmark-only \
            --batch-size 1 \
            --socket-id 0 \
            --docker-image gcr.io/deeplearning-platform-release/tf-cpu.1-14 \
            --in-graph $MODEL_WORK_DIR/wide_deep_fp32_pretrained_model.pb \
            --data-location $MODEL_WORK_DIR/eval_preprocessed_eval.tfrecords \
            -- num_parallel_batches=1
        ```
    * Running in batch inference mode, set `--batch-size 512`
        ```
        $ cd $MODEL_WORK_DIR/models/benchmarks

        $ python launch_benchmark.py \
            --model-name wide_deep_large_ds \
            --precision fp32 \
            --mode inference \
            --framework tensorflow \
            --benchmark-only \
            --batch-size 512 \
            --socket-id 0 \
            --docker-image gcr.io/deeplearning-platform-release/tf-cpu.1-14 \
            --in-graph $MODEL_WORK_DIR/wide_deep_fp32_pretrained_model.pb \
            --data-location $MODEL_WORK_DIR/eval_preprocessed_eval.tfrecords
        ```
    * The log file is saved to the value of `--output-dir`. The tail of the log output when the script completes 
        should look something like this:
        ```
        --------------------------------------------------
        Total test records           :  2000000
        No of correct predictions    :  1550447
        Batch size is                :  512
        Number of batches            :  1954
        Classification accuracy (%)  :  77.5223
        Inference duration (seconds) :  3.4977
        Latency (millisecond/batch)  :  0.001749
        Throughput is (records/sec)  :  571802.228
        --------------------------------------------------
        Ran inference with batch size 512
        Log location outside container: {--output-dir value}/benchmark_wide_deep_large_ds_inference_fp32_20190225_062206.log
        ```

4. To return to where you started from:
```
$ popd
```
