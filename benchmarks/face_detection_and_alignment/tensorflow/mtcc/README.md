# MTCC

This document has instructions for how to run MTCC for the
following modes/precisions:
* [FP32 inference](#fp32-inference-instructions)

Instructions for MTCC model training and inference for other precisions are coming later.

## FP32 Inference Instructions

1. Store path to current directory and clone the MTCC model repository [AITTSMD/MTCNN-Tensorflow](https://github.com/AITTSMD/MTCNN-Tensorflow):
```
    $ MODEL_WORK_DIR=${MODEL_WORK_DIR:=`pwd`}
    $ pushd $MODEL_WORK_DIR
    
    $ git clone https://github.com/AITTSMD/MTCNN-Tensorflow.git
```

2. Download and extract the [dataset](http://mmlab.ie.cuhk.edu.hk/archive/CNN/data/train.zip), `lfw_5590` will be used. 
```
    $ wget http://mmlab.ie.cuhk.edu.hk/archive/CNN/data/train.zip
    $ unzip train.zip
    $ ls -l
    drwxr-xr-x  5592 <user> <group>      178944 Apr 15  2013 lfw_5590
    drwxr-xr-x  7878 <user> <group>      252096 Apr 15  2013 net_7876
    -rw-r--r--     1 <user> <group>      519406 Apr 15  2013 testImageList.txt
    -rw-r--r--@    1 <user> <group>   136492573 Mar 22 11:54 train.zip
    -rw-r--r--     1 <user> <group>     1498353 Apr 15  2013 trainImageList.txt
```

3. Download the pre-trained model.
```
    $ wget https://storage.googleapis.com/intel-optimized-tensorflow/models/mtcc_fp32_pretrained_model.tar.gz
    $ tar -xzvf mtcc_fp32_pretrained_model.tar.gz
```

4. Clone the [intelai/models](https://github.com/intelai/models) repo.
This repo has the launch script for running models.

```
    $ git clone https://github.com/IntelAI/models.git
    Cloning into 'models'...
```

5. Run the `launch_benchmark.py` script from the intelai/models repo with the appropriate parameters including: the `--model-source-dir` from step 1, `--data-location` from step 2,
and the `--checkpoint` from step 3.

Run:
```
    $ cd models/benchmarks
    
    $ python launch_benchmark.py \
        --data-location $MODEL_WORK_DIR/lfw_5590 \
        --model-source-dir $MODEL_WORK_DIR/MTCNN-Tensorflow \
        --model-name mtcc \
        --framework tensorflow \
        --precision fp32 \
        --mode inference \
        --socket-id 0 \
        --checkpoint $MODEL_WORK_DIR/MTCNN_model \
        --docker-image gcr.io/deeplearning-platform-release/tf-cpu.1-14
```

6. The log file is saved to the value of `--output-dir`.

Below is a sample log file tail when running for batch inference, online inference, and accuracy:

```
time cost 0.459  pnet 0.166  rnet 0.144  onet 0.149
time cost 0.508  pnet 0.034  rnet 0.010  onet 0.005
time cost 0.548  pnet 0.028  rnet 0.008  onet 0.005
time cost 0.585  pnet 0.025  rnet 0.007  onet 0.005
time cost 0.627  pnet 0.028  rnet 0.009  onet 0.005
...
time cost 220.844  pnet 0.027  rnet 0.008  onet 0.005
time cost 220.884  pnet 0.027  rnet 0.008  onet 0.005
Accuracy: 1.12
Total images: 5590
Latency is: 40.36, Throughput is: 24.78
Ran inference with batch size -1
Log location outside container: /home/user/models/benchmarks/common/tensorflow/logs/benchmark_mtcc_inference_fp32_20190322_221543.log
```

7. To return to where you started from:
```
$ popd
```