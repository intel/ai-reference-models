# Mozilla DeepSpeech

This document has instructions for how to run Mozilla DeepSpeech benchmark for the
following modes/platforms:
* [FP32 inference](#fp32-inference-instructions)

Benchmarking instructions and scripts for model training and inference for
other platforms are coming later.

## FP32 Inference Instructions

1. Clone the [intelai/models](https://github.com/intelai/models) repo.
This repo has the launch script for running benchmarking.

    ```
    $ git clone git@github.com:IntelAI/models.git
    ```
    
2. Clone Mozilla DeepSpeech and tensorflow git repositories
    
    Make sure you have `git-lfs` installed before cloning DeepSpeech repo,
    otherwise `data/lm/lm.binary` (~1.8GB) file won't be downloaded
    
    ```
    $ mkdir mozilla
    $ cd mozilla
    $ git clone https://github.com/mozilla/DeepSpeech.git
    $ git clone https://github.com/mozilla/tensorflow.git
    ```
    
    Verify that in DeepSpeech repo `data/lm/lm.binary` file size is ~1.8GB.

3. Download pre-trained model checkpoint files    

   ``` 
   $ wget https://github.com/mozilla/DeepSpeech/releases/download/v0.3.0/deepspeech-0.3.0-checkpoint.tar.gz
   $ mkdir -p checkpoints && tar -C ./checkpoints/ -zxf deepspeech-0.3.0-checkpoint.tar.gz
   ```
4. Download dataset

   ```
   $ wget https://github.com/mozilla/DeepSpeech/releases/download/v0.3.0/audio-0.3.0.tar.gz
   $ tar -C /home/myuser/data -xvf audio-0.3.0.tar.gz
   ``` 
5. How to run benchmarks

    Make sure `datafile_name` is relative path from `--data-location`.

    ```
    python launch_benchmark.py 
        --framework tensorflow 
        --platform fp32 
        --mode inference 
        --model-name deep-speech 
        --batch-size 1 
        --data-location /home/myuser/data 
        --checkpoint /home/myuser/checkpoints 
        --docker-image intelaipg/intel-optimized-tensorflow:latest-devel-mkl 
        --model-source-dir /home/user/mozilla
        --single-socket 
        --verbose
        -- datafile_name=audio/8455-210777-0068.wav
    ``` 

6. The log file is saved to:
`models/benchmarks/common/tensorflow/logs/`

   The tail of the log output when the benchmarking completes should look
   something like this:
   
   ```
    Installing collected packages: numpy, ds-ctcdecoder
    Successfully installed ds-ctcdecoder-0.4.0a0 numpy-1.15.4
    he wanted to go by home and tell his wife and children good by and get his clothes it was no go
    lscpu_path_cmd = command -v lscpu
    lscpu located here: /usr/bin/lscpu
    Running: numactl --physcpubind=0-17 --membind=0 taskset -c 0-17 python -u DeepSpeech.py --log_level 1 --checkpoint_dir "/checkpoints" --one_shot_infer "/dataset/5694-64038-0019.wav" --inter_op 1 --intra_op 18 --num_omp_threads 18
    Ran inference with batch size 1
    Log location outside container: /home/myuser/intel-models/benchmarks/common/tensorflow/logs/benchmark_deep-speech_inference_fp32_20181129_174240.log
  ```
   