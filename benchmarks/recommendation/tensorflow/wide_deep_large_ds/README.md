# Wide & Deep

This document has instructions for how to run Wide & Deep for the
following modes/precisions:

* [Prepare dataset](#Prepare-dataset)
* [INT8 inference](#int8-inference-instructions)
* [FP32 inference](#fp32-inference-instructions)
* [FP32 Training](#fp32-training-instructions)

## Prepare dataset

1. Store the path to the current directory and clone the [intelai/models](https://github.com/intelai/models) repo.
    ```
    $ mkdir wide_deep_large_ds
    $ cd wide_deep_large_ds
    $ MODEL_WORK_DIR=${MODEL_WORK_DIR:=`pwd`}
    $ pushd $MODEL_WORK_DIR

    $ git clone https://github.com/IntelAI/models.git
    ```

2. Download large Kaggle Display Advertising Challenge Dataset 

   Download large Kaggle Display Advertising Challenge Dataset from http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/

   The evaluation dataset for accuracy measurement is not available in the above link can be downloaded from https://storage.googleapis.com/dataset-uploader/criteo-kaggle/large_version/eval.csv

   Download the train dataset(in csv format) from https://storage.googleapis.com/dataset-uploader/criteo-kaggle/large_version/train.csv
3. Pre-process the downloaded dataset to tfrecords using [preprocess_csv_tfrecords.py](/models/recommendation/tensorflow/wide_deep_large_ds/dataset/preprocess_csv_tfrecords.py)

    Copy the eval.csv and test.csv into your current working directory (i.e. root of models repo) and launch. This preprocess step requires Pandas module to be installed.

    * Launch docker 
        ```
        $ cd $MODEL_WORK_DIR/models/

        $ docker run -it --privileged -u root:root \
                    -w /models \
                    --volume $MODEL_WORK_DIR:/models \
                    intel/intel-optimized-tensorflow:1.15.2 \
                   /bin/bash

        ```
    * Process eval dataset, `pandas` is a dependency in `preprocess_csv_tfrecords.py`, please install it inside the container:
        ```
        apt-get install python-pandas
        pip install pandas
        ```
        
   * If you are unable to fetch please do:
        ```
        apt-get update
        
        ```
    * Now run the data preprocessing step:
        ```
        cd models
        python models/recommendation/tensorflow/wide_deep_large_ds/dataset/preprocess_csv_tfrecords.py \
                --inputcsv-datafile eval.csv \
                --calibrationcsv-datafile train.csv \
                --outputfile-name preprocessed_eval
        ```
    
    * Process Train dataset for Model Quantization
        ```
        python models/recommendation/tensorflow/wide_deep_large_ds/dataset/preprocess_csv_tfrecords.py \
                --inputcsv-datafile train.csv \
                --calibrationcsv-datafile eval.csv \
                --outputfile-name preprocessed_train
        ```    
    * Process test dataset

        If you have test dataset without true labels, following command can be used to generate the processed test set on which you can run inference.
        The test.txt is in tab-separated values (TSV) format and they must be converted into comma-separated values (CSV) format before doing the pre-processing. On docker console run the below commands to pre-process test datasets
        ```
        $ tr '\t' ',' < test.txt > test.csv
        python models/recommendation/tensorflow/wide_deep_large_ds/dataset/preprocess_csv_tfrecords.py \
                --inputcsv-datafile test.csv \
                --outputfile-name preprocessed_test
        ```
        Now preprocessed eval, train and test datasets will be stored as eval_preprocessed_eval.tfrecords , train_preprocessed_train.tfrecords and test_preprocessed_test.tfrecords respectively in $MODEL_WORK_DIR/models/ directory

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

    $ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/wide_deep_int8_pretrained_model.pb
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
            --accuracy-only \
            --docker-image intel/intel-optimized-tensorflow:1.15.2 \
            --in-graph $MODEL_WORK_DIR/wide_deep_int8_pretrained_model.pb \
            --data-location $MODEL_WORK_DIR/models/eval_preprocessed_eval.tfrecords
        ```

3. Run Performance test


   * Running online inference mode to measure latency, set `--batch-size 1`
     
       ``` 
       $ cd $MODEL_WORK_DIR/models/benchmarks

       $ python launch_benchmark.py \
            --model-name wide_deep_large_ds \
            --precision int8 \
            --mode inference \
            --framework tensorflow \
            --benchmark-only \
            --batch-size 1 \
            --docker-image intel/intel-optimized-tensorflow:1.15.2 \
            --in-graph $MODEL_WORK_DIR/wide_deep_int8_pretrained_model.pb \
            --data-location $MODEL_WORK_DIR/models/eval_preprocessed_eval.tfrecords \
            --num-intra-threads 1 --num-inter-threads 1  \
            -- num_omp_threads=1
       ```
   * Running batch inference mode, set `--batch-size 512` \
        The "numactl" is a utility which can be used to control NUMA policy for processes or shared memory. To install numactl do:
        ```
        apt install numactl
        ``` 
        By default numactl is disabled. User can use NUMA policy control only on bare metal in advanced cases to specify number of cores to be used. This can be specified as shown in the command below only on bare metal. Below are the commands that gives best performance on 28 cores. The hyperparameters  num-intra-threads, num-inter-threads, num_omp_threads etc should be tuned for best performance. 

        Case 1 : Disabling `use_parallel_batches` option. In this case the batches are inferred in sequential order. By default `use_parallel_batches` is disabled. Kmp variables can also be set by using the arguments shown below or through config file $MODEL_WORK_DIR/models/benchmarks/recommendation/tensorflow/wide_deep_large_ds/inference/int8/config.json


           ``` 
            $ cd $MODEL_WORK_DIR/models/benchmarks
        
            $ numactl --physcpubind=0-27 -m 0 python launch_benchmark.py \
                --model-name wide_deep_large_ds \
                --precision int8 \
                --mode inference \
                --framework tensorflow \
                --benchmark-only \
                --batch-size 512 \
                --in-graph $MODEL_WORK_DIR/wide_deep_int8_pretrained_model.pb \
                --data-location $MODEL_WORK_DIR/models/eval_preprocessed_eval.tfrecords \
                --num-intra-threads 28 --num-inter-threads 1 \
                -- num_omp_threads=16  kmp_block_time=1 kmp_settings=1 
           ```
           * The log file is saved to the value of `--output-dir`. The tail of the log output when the script completes 
             should look something like this:

                ```
                --------------------------------------------------
                Total test records           :  2000000
                Batch size is                :  512
                Number of batches            :  3907
                Classification accuracy (%)  :  77.636
                Inference duration (seconds) :  3.1534
                Average Latency (ms/batch)   :  0.8285
                Throughput is (records/sec)  :  617992.762 
                --------------------------------------------------

                Ran inference with batch size 512
                Log location outside container:  {--output-dir value}/benchmark_wide_deep_large_ds_inference_int8_20190225_061815.log
                ```


        Case 2 : Enabling `use_parallel_batches` option. In this case multiples batches are inferred in parallel. Number of batches to be executed in parallel can be given by argument num_parallel_batches.

        ``` 
            $ cd $MODEL_WORK_DIR/models/benchmarks
        
            $ numactl --physcpubind=0-27 -m 0 python launch_benchmark.py \
                --model-name wide_deep_large_ds \
                --precision int8 \
                --mode inference \
                --framework tensorflow \
                --benchmark-only \
                --batch-size 512 \
                --in-graph $MODEL_WORK_DIR/wide_deep_int8_pretrained_model.pb \
                --data-location $MODEL_WORK_DIR/models/eval_preprocessed_eval.tfrecords \
                --num-intra-threads 1 --num-inter-threads 28  \
                -- num_omp_threads=1 use_parallel_batches=True num_parallel_batches=28 kmp_block_time=0 kmp_settings=1 
           ```

           * The log file is saved to the value of `--output-dir`. The tail of the log output when the script completes 
             should look something like this:

                ```
                --------------------------------------------------
                Total test records           :  2000000
                Batch size is                :  512
                Number of batches            :  3907
                Inference duration (seconds) :  1.7056
                Average Latency (ms/batch)   :  12.2259
                Throughput is (records/sec)  :  1172597.21 
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

    $wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/wide_deep_fp32_pretrained_model.pb
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
            --accuracy-only \
            --docker-image intel/intel-optimized-tensorflow:2.3.0 \
            --in-graph $MODEL_WORK_DIR/wide_deep_fp32_pretrained_model.pb \
            --data-location $MODEL_WORK_DIR/models/eval_preprocessed_eval.tfrecords
       ```

3. Run Performance test

    * Running in online inference mode for measuring latency , set `--batch-size 1`

        ```
        $ cd $MODEL_WORK_DIR/models/benchmarks

        $ python launch_benchmark.py \
            --model-name wide_deep_large_ds \
            --precision fp32 \
            --mode inference \
            --framework tensorflow \
            --benchmark-only \
            --batch-size 1 \
            --docker-image intel/intel-optimized-tensorflow:2.3.0 \
            --in-graph $MODEL_WORK_DIR/wide_deep_fp32_pretrained_model.pb \
            --data-location $MODEL_WORK_DIR/models/eval_preprocessed_eval.tfrecords \
            --num-intra-threads 1 --num-inter-threads 1  \
            -- num_omp_threads=1
        ```
    * Running in batch inference mode, set `--batch-size 512`
        By default numactl is disabled. User can specify as shown in the command below. Below are the commands that gives best performance on 28 cores. The hyperparameters  num-intra-threads, num-inter-threads, num_omp_threads etc should be tuned for best performance.
        
        Case 1 : Disabling `use_parallel_batches` option. Kmp variables can also be set by using the arguments shown below or through config file $MODEL_WORK_DIR/models/benchmarks/recommendation/tensorflow/wide_deep_large_ds/inference/fp32/config.json

            ```
            $ cd $MODEL_WORK_DIR/models/benchmarks

            $ numactl --physcpubind=0-27 -m 0 python launch_benchmark.py \
                --model-name wide_deep_large_ds \
                --precision fp32 \
                --mode inference \
                --framework tensorflow \
                --benchmark-only \
                --batch-size 512 \
                --in-graph $MODEL_WORK_DIR/wide_deep_fp32_pretrained_model.pb \
                --data-location $MODEL_WORK_DIR/models/eval_preprocessed_eval.tfrecords \
                --num-intra-threads 28 --num-inter-threads 1  \
                -- num_omp_threads=20 kmp_block_time=1 kmp_settings=1 
            ```
                --------------------------------------------------
                Total test records           :  2000000
                Batch size is                :  512
                Number of batches            :  3907
                Classification accuracy (%)  :  77.6693
                No of correct predictions    :  1553386
                Inference duration (seconds) :  6.4442
                Average Latency (ms/batch)   :  1.6931
                Throughput is (records/sec)  :  302410.809  
                --------------------------------------------------
         
        Case 2 : Enabling `use_parallel_batches` option.

            ```
            cd $MODEL_WORK_DIR/models/benchmarks

            numactl --physcpubind=0-27 -m 0 python launch_benchmark.py \
                --model-name wide_deep_large_ds \
                --precision fp32 \
                --mode inference \
                --framework tensorflow \
                --benchmark-only \
                --batch-size 512 \
                --in-graph $MODEL_WORK_DIR/wide_deep_fp32_pretrained_model.pb \
                --data-location $MODEL_WORK_DIR/models/eval_preprocessed_eval.tfrecords \
                --num-intra-threads 1 --num-inter-threads 28  \
                -- num_omp_threads=1 use_parallel_batches=True num_parallel_batches=28 kmp_block_time=0 kmp_settings=1
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
                Inference duration (seconds) :  3.4655
                Average Latency (ms/batch)   :  24.8406
                Throughput is (records/sec)  :  577120.456 
                --------------------------------------------------
                Ran inference with batch size 512
                Log location outside container: {--output-dir value}/benchmark_wide_deep_large_ds_inference_fp32_20190225_062206.log
                ```


4. To return to where you started from:
```
$ popd
```
## FP32 Training Instructions

1. Download large Kaggle Display Advertising Challenge Dataset 

   Download large Kaggle Display Advertising Challenge Dataset from http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/ 

   The evaluation dataset for accuracy measurement is not available in the above link can be downloaded from https://storage.googleapis.com/dataset-uploader/criteo-kaggle/large_version/eval.csv

   Download the large version of train dataset(in csv format) from https://storage.googleapis.com/dataset-uploader/criteo-kaggle/large_version/train.csv
   
2. Train Wide and Deep model by providing location of train.csv, eval.csv 

    * Train the model (The model will be trained for 10 epochs if -- steps is not specified). --output-dir arg can be used to specify the directory where checkpoints and saved model to be saved.
        ```
        cd $MODEL_WORK_DIR/models/benchmarks
        
        $ python launch_benchmark.py --model-name wide_deep_large_ds \
           --precision fp32 \
           --mode training  \
           --framework tensorflow \
           --batch-size 512 \
           --data-location $MODEL_WORK_DIR \
           --docker-image intel/intel-optimized-tensorflow:2.3.0
        
        ```
    Once the training completes successfully the path of checkpoint files and saved_model.pb will be printed as shown below
    ```
    INFO:tensorflow:SavedModel written to: home/<user>/temp-1602670603/saved_model.pb
    Using TensorFlow version 2.3.0
    Begin training and evaluation
    Saving model checkpoints to /home/<user>/model_WIDE_AND_DEEP_1602670581
    ****Computing statistics of train dataset*****
    estimator built
    fit done
    evaluate done
    Model exported to home/<user>
    ```

    