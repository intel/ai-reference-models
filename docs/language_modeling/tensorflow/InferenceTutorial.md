# Language Modeling Inference with BERT Large

## Goal
This tutorial will introduce CPU performance considerations for the deep learning model BERT Large for language modeling and demonstrate how to use Intel® Optimizations for TensorFlow to improve inference time on CPUs.
This tutorial will also provide code examples to use Intel Model Zoo's pre-trained BERT model for a quick off-the-ground implementation.

## Background
With BFloat16 (BF16) instructions and optimizations now in the Intel® Xeon® Scalable processor and Intel® Optimizations for TensorFlow, deep learning workload performance can benefit from a smaller data representation (16-bit instead of the traditional 32-bit floating point) often with little or no loss of accuracy.
This is because the BF16 standard halves the data size in a way that retains most of the precision near zero while sacrificing more precision at the extremes of the numerical range. For many machine and deep learning tasks, this is a favorable trade-off.
For more technical details, see this article on [lowering numerical precision to increase deep learning performance](https://www.intel.com/content/www/us/en/artificial-intelligence/posts/lowering-numerical-precision-increase-deep-learning-performance.html).

BERT (Bidirectional Encoder Representations from Transformers) is a popular language modeling topology.
Since its [publication](https://arxiv.org/pdf/1810.04805.pdf) in May 2019, BERT has quickly become state-of-the-art for many Natural Language Processing (NLP) tasks, including question answering and next sentence prediction.
The BERT Large variant has 340 million parameters and uses an innovative masked language model (MLM) pre-training approach that allows a second training stage called fine-tuning to achieve a wide variety of NLP tasks.
To demonstrate Bert Large inference performance with BF16 precision, this tutorial uses the Intel Model Zoo's BERT Large pre-trained model which has been fine-tuned for question answering with the [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) dataset.
The tutorial concludes with FP32 inference for comparison of performance and accuracy.

##  Recommended Settings
In addition to TensorFlow optimizations that use the [Intel® oneAPI Deep Neural Network Library (Intel® oneDNN)](https://github.com/oneapi-src/oneDNN), the run-time settings also significantly contribute to improved performance.
Tuning these options to optimize CPU workloads is vital to optimize performance of TensorFlow on Intel® processors.
Below are the set of run-time options tested empirically on BERT Large and recommended by Intel:

| Run-time options  | Recommendations |
| ------------- | ------------- |
| Batch Size | 32. Regardless of the hardware  |
| Hyperthreading  | Enabled. Turn on in BIOS. Requires a restart. |
|intra_op_parallelism_threads |# physical cores |
|inter_op_parallelism_threads | 1 or 2|
|NUMA Controls| --cpunodebind=0 --membind=0 |
|KMP_AFFINITY| KMP_AFFINITY=granularity=fine,verbose,compact,1,0|
|KMP_BLOCKTIME| 1 |
|KMP_SETTINGS| 1 |
|OMP_NUM_THREADS |# physical cores - 1 or # physical cores - 2|

Note 1: [Refer](https://www.intel.com/content/www/us/en/developer/articles/technical/maximize-tensorflow-performance-on-cpu-considerations-and-recommendations-for-inference.html) to learn more about the run-time options.

Note 2: You can remove `verbose` from `KMP_AFFINITY` setting to avoid verbose output at runtime.

Run the following commands to get your processor information:

a. # physical cores per socket: `lscpu | grep "Core(s) per socket" | cut -d':' -f2 | xargs`

b. # all physical cores: `lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l`

Below is a code snippet you can incorporate into your existing TensorFlow application to set the best settings.
You can either set them in the CLI or in the Python script. Note that inter and intra_op_parallelism_threads settings can only be set
in the Python script.

```bash
export OMP_NUM_THREADS=<# physical cores - 2>
export KMP_AFFINITY="granularity=fine,verbose,compact,1,0"
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
```
(or)
```
import os
os.environ["KMP_BLOCKTIME"] = "1"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
os.environ["OMP_NUM_THREADS"]= <# physical cores - 2>
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(<# physical cores>)
```

## Hands-on Tutorial
This section shows how to measure and compare BF16 and FP32 inference performance on Intel's Model Zoo pre-trained model (or your pre-trained model) by setting the above-discussed run-time flags.

### Initial Setup

Note: These steps are adapted from the BERT Large Inference [README](/benchmarks/language_modeling/tensorflow/bert_large/README.md#inference-instructions).
Please check there for the most up-to-date information and links.

1. Clone IntelAI models and download into your home directory, skip this step if you already have Intel AI models installed.

```bash
cd ~
git clone https://github.com/IntelAI/models.git
```

2. Download and unzip the BERT large uncased (whole word masking) model from the [google bert repo](https://github.com/google-research/bert#pre-trained-models).
Then, download the `dev-v1.1.json` file from the [google bert repo](https://github.com/google-research/bert#squad-11) into the `wwm_uncased_L-24_H-1024_A-16` directory that was just unzipped.

```
wget https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip
unzip wwm_uncased_L-24_H-1024_A-16.zip

wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -P wwm_uncased_L-24_H-1024_A-16
```
The `wwm_uncased_L-24_H-1024_A-16` directory is what will be passed as the `--data-location` when running inference.

3. Download and unzip the pre-trained model. The file is 3.4GB so it will take some time.

```
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/bert_large_checkpoints.zip
unzip bert_large_checkpoints.zip
```
This directory will be passed as the `--checkpoint` location when running inference.

4. Install [Docker](https://docs.docker.com/v17.09/engine/installation/) since the tutorial runs in a Docker container.

5. Pull the relevant Intel-optimized TensorFlow Docker image.
Click here to find  all the [available](https://www.intel.com/content/www/us/en/developer/articles/guide/optimization-for-tensorflow-installation-guide.html) docker images.
```bash
docker pull intel/intel-optimized-tensorflow:latest
```

6. Navigate to the inference script directory in local IntelAI repository.
```bash
cd ~/models/benchmarks
```

### BF16 Inference

Run the Python script `launch_benchmark.py` with the pre-trained model.
The `launch_benchmark.py` script can be treated as an entry point to conveniently perform out-of-box high performance inference on pre-trained models from the Intel Model Zoo.
The script will automatically set the recommended run-time options for supported topologies, but if you choose to set your own options, refer to the full list of available flags and a detailed explanation of [`launch_benchmark.py`](/docs/general/tensorflow/LaunchBenchmark.md).
This step will automatically launch a new container on every run and terminate. Go to [this optional step](#optional) to interactively run the script on the container.

1. <b>*BF16 Batch Inference*</b>

Console in:
```bash
python launch_benchmark.py \
    --model-name=bert_large \
    --precision=bfloat16 \
    --mode=inference \
    --framework=tensorflow \
    --batch-size=32 \
    --data-location ~/wwm_uncased_L-24_H-1024_A-16 \
    --checkpoint ~/bert_large_checkpoints \
    --output-dir ~/output \
    --benchmark-only \
    --docker-image intel/intel-optimized-tensorflow:latest \
    -- infer_option=SQuAD
```
Console out:
```
...
I0424 21:14:28.002666 140184442087232 run_squad.py:1365] Processed #examples: 960
INFO:tensorflow:prediction_loop marked as finished
Elapsed time: ...
throughput((num_processed_examples-threshod_examples)/Elapsedtime): ...
Ran inference with batch size 32
Log location outside container: /~/output/benchmark_bert_large_inference_bfloat16_20200424_210607.log
```

2. <b>*BF16 Accuracy*</b>

Console in:
```bash
python launch_benchmark.py \
    --model-name=bert_large \
    --precision=bfloat16 \
    --mode=inference \
    --framework=tensorflow \
    --batch-size=32 \
    --data-location ~/wwm_uncased_L-24_H-1024_A-16 \
    --checkpoint ~/bert_large_checkpoints \
    --output-dir ~/output \
    --accuracy-only \
    --docker-image intel/intel-optimized-tensorflow:latest \
    -- infer_option=SQuAD
```

Console out:
```bash
INFO:tensorflow:Processing example: 10830
I0428 00:26:11.595798 140332503766848 run_squad.py:1370] Processing example: 10830
INFO:tensorflow:prediction_loop marked as finished
INFO:tensorflow:Writing predictions to: /~/output/predictions.json
I0428 00:26:11.794145 140332503766848 run_squad.py:804] Writing predictions to: /~/output/predictions.json
INFO:tensorflow:Writing nbest to: /~/output/nbest_predictions.json
I0428 00:26:11.794228 140332503766848 run_squad.py:805] Writing nbest to: /~/output/nbest_predictions.json
{"exact_match": ..., "f1": ...}
Ran inference with batch size 32
Log location outside container: /~/output/benchmark_bert_large_inference_bfloat16_20200427_224428.log
```

Output files and logs are saved to the `--output-dir` or to the default location models/benchmarks/common/tensorflow/logs, if no `--output-dir` is set.

### FP32 Inference

1. <b>*FP32 Batch Inference*</b>

To see the FP32 batch inference performance, run the same command from above but change `--precision=bfloat16` to `--precision=fp32`.

```bash
python launch_benchmark.py \
    --model-name=bert_large \
    --precision=fp32 \
    --mode=inference \
    --framework=tensorflow \
    --batch-size=32 \
    --data-location ~/wwm_uncased_L-24_H-1024_A-16 \
    --checkpoint ~/bert_large_checkpoints \
    --output-dir ~/output \
    --benchmark-only \
    --docker-image intel/intel-optimized-tensorflow:latest \
    -- infer_option=SQuAD
```

2. <b>*FP32 Accuracy*</b>

Similarly, to see the FP32 accuracy, run the above command but change `--precision=bfloat16` to `--precision=fp32`.

```bash
python launch_benchmark.py \
    --model-name=bert_large \
    --precision=fp32 \
    --mode=inference \
    --framework=tensorflow \
    --batch-size=32 \
    --data-location ~/wwm_uncased_L-24_H-1024_A-16 \
    --checkpoint ~/bert_large_checkpoints \
    --output-dir ~/output \
    --accuracy-only \
    --docker-image intel/intel-optimized-tensorflow:latest \
    -- infer_option=SQuAD
```

### Interactive Option

<a name="optional"></a>If you want to run `launch_benchmark.py` interactively from within the docker container, add flag `--debug`. This will launch a docker container based on the `--docker_image`,
perform necessary installs, and run the `launch_benchmark.py` script, but does not terminate the container process. As an example, this is how you would launch interactive BF16 batch inference for benchmarking:

Console in:
```bash
python launch_benchmark.py \
    --model-name=bert_large \
    --precision=bfloat16 \
    --mode=inference \
    --framework=tensorflow \
    --batch-size=32 \
    --data-location ~/wwm_uncased_L-24_H-1024_A-16 \
    --checkpoint ~/bert_large_checkpoints \
    --output-dir ~/output \
    --benchmark-only \
    --docker-image intel/intel-optimized-tensorflow:latest \
    --debug \
    -- infer_option=SQuAD
```

Console out:
```bash
root@c49f3442efb1:/workspace/benchmarks/common/tensorflow#
```

To rerun the benchmarking script, execute the `start.sh` bash script from your existing directory with the available flags, which in turn will run `launch_benchmark.py`.
For example, to run with different batch size settings (e.g. batch size=64) run with `BATCH_SIZE`
and to skip the run from reinstalling packages pass `True` to `NOINSTALL`.

```bash
chmod +x ./start.sh
NOINSTALL=True BATCH_SIZE=64 ./start.sh
```

All other flags will be defaulted to values passed in the first `launch_benchmark.py` that starts the container. [See here](/docs/general/tensorflow/LaunchBenchmark.md) to get the full list of flags.
