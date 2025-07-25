# Language Modeling BFloat16 Training with BERT Large on Multi-node Intel CPUs

## Goal

This tutorial will introduce CPU performance considerations for training the popular [BERT](https://arxiv.org/abs/1810.04805) model with BFloat16 (BF16) data type on CPUs, and how to use Intel® Optimizations for TensorFlow\* to improve training time.

## Background

The release of BERT represents a new era in Natural Language Processing (NLP). BERT stands for Bidirectional Encoder Representations from Transformers, which is a new language representation model for NLP pre-training developed by Google. BERT is open source and available in [GitHub](https://github.com/google-research/bert). It is conceptually simple and empirically powerful and it demonstrates how well the Deep Learning (DL) model can process a wide range of language-based tasks such as question answering and language inference.

By default, TensorFlow stores all variables in 32-bit floating-point (FP32) precision. However, many DL models in the image classification, speech recognition, language modeling, generative networks, and industrial recommendation systems fields can use lower precision data types such as the [Brain Floating Point (BF16)](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format) half-precision format without significant accuracy degradation to speed up the execution time and decrease the memory usage. BF16 is popular and attractive for DL training as it can represent the same range of values as that of IEEE 754 floating-point format (FP32) and it is very simple to convert to/from FP32.

The BF16 format is supported in popular DL frameworks such as TensorFlow\* and PyTorch\* and is accelerated in hardware on platforms such as the [3rd Generation Intel® Xeon® Scalable Processors](https://ark.intel.com/content/www/us/en/ark/products/series/204098/3rd-generation-intel-xeon-scalable-processors.html), [Intel®
 Agilex™ FPGAs](https://www.intel.com/content/dam/www/programmable/us/en/pdfs/literature/wp/intel-agilex-fpgas-deliver-game-changing-combination-wp.pdf), and [Google\* Cloud Tensor Processing Units (TPUs)](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus).

## Recommended settings

In addition to TensorFlow\* optimizations that use the Intel® oneAPI Deep Neural Network Library (Intel® oneDNN), the run-time settings also significantly contribute to improved performance.
Tuning these options is key to the performance of TensorFlow\* on 3rd Generation Intel® Xeon® Scalable Processors.
Below are the set of run-time options tested empirically on BERT Large and recommended by Intel:

| Run-time options  | Recommendations |
| ------------- | ------------- |
| Batch Size | 24  |
| Hyperthreading  | Enabled. Turn on in BIOS. Requires a restart. |
|intra_op_parallelism_threads |# physical cores | 
|inter_op_parallelism_threads | 1 |
|NUMA Controls| --cpunodebind=0 --membind=0 |
|KMP_AFFINITY| KMP_AFFINITY=granularity=fine,verbose,compact,1,0|
|KMP_BLOCKTIME| 1 |
|OMP_NUM_THREADS |physical cores|
 
Note 1: [Refer](https://www.intel.com/content/www/us/en/developer/articles/technical/maximize-tensorflow-performance-on-cpu-considerations-and-recommendations-for-inference.html) to learn more about the run time options.

Note 2: You can remove `verbose` from `KMP_AFFINITY` setting to avoid verbose output at runtime. 

Run the following commands to get your processor information:

a. #physical cores per socket : `lscpu | grep "Core(s) per socket" | cut -d':' -f2 | xargs`

b. #all physical cores: `lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l`

Below is a code snippet you can incorporate into your existing TensorFlow application to set the best settings. 
You can either set them in the CLI or in the Python script. Note that inter and intra_op_parallelism_threads settings can only be set 
in the Python script.

```bash
export OMP_NUM_THREADS=physical cores
export KMP_AFFINITY="granularity=fine,verbose,compact,1,0"
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
```

(or)

```python
import os
os.environ["KMP_BLOCKTIME"] = "1"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
os.environ["OMP_NUM_THREADS"]= <# physical cores>
tf.config.threading.set_intra_op_parallelism_threads(<# physical cores>)
tf.config.threading.set_inter_op_parallelism_threads(1)
```

## Hands-on Tutorial

This section shows how to run the BF16 BERT training on 3rd Generation Intel® Xeon® Scalable Processors.

### Initial Setup

Note: These steps are adapted from the BERT Large Inference [README](/benchmarks/language_modeling/tensorflow/bert_large/README.md#inference-instructions).
Please check there for the most up-to-date information and links.

1. Clone the Model Zoo for Intel® Architecture into your home directory; skip this step if you already have this repository cloned.

```bash
cd ~
git clone https://github.com/IntelAI/models.git
```

2. Download and unzip the BERT large uncased (whole word masking) pre-trained model from the [Google's\* BERT Github\* repository](https://github.com/google-research/bert#pre-trained-models). Export the environment variable `BERT_LARGE_DIR` to the unzipped BERT large uncased model directory.

```bash
wget https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip
unzip wwm_uncased_L-24_H-1024_A-16.zip
export BERT_LARGE_DIR=$PWD/wwm_uncased_L-24_H-1024_A-16
```

3. Create a SQUAD directory, download the `train-v1.1.json` and `dev-v1.1.json` files from the [Google\* BERT Github\* repository's](https://github.com/google-research/bert#squad-11) SQuAD 1.1 section into this directory and export the `SQUAD_DIR` environment variable to point at the SQUAD directory:

```bash
mkdir SQUAD && cd SQUAD
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
wget https://raw.githubusercontent.com/allenai/bi-att-flow/master/squad/evaluate-v1.1.py
export SQUAD_DIR=$PWD
```

4. Install [Docker](https://docs.docker.com/v17.09/engine/installation/) since the tutorial runs in a Docker container. 

5. Pull the Intel® Optimization for TensorFlow\* Docker\* image that contains BF16 optimizations. 
[View the Intel® Optimization for TensorFlow\* Docker\* landing page](https://hub.docker.com/r/intel/intel-optimized-tensorflow) to find other available Docker images.

```bash
docker pull intel/intel-optimized-tensorflow:latest
```

The Docker\* image has enabled OpenMPI\* and Horovod\*.

6. Navigate to the benchmark directory in local Model Zoo for Intel® Architecture repository.

```bash        
cd ~/models/benchmarks
```

### BF16 Multi-node Training

To train the BERT model, run the Python\* script `launch_benchmark.py`. 
The `launch_benchmark.py` script can be treated as an entry point to conveniently perform out-of-box high performance inference on pre-trained models from the Model Zoo for Intel® Architecture. 
The script will automatically set the recommended run-time options for supported topologies, but if you choose to set your own options, refer to the [full list of available flags and a detailed explanation of `launch_benchmark.py`](/docs/general/tensorflow/LaunchBenchmark.md).

To run BF16 BERT distributed training for different tasks: SQuAD, Classifiers and Pretraining (e.g. one MPI process per socket) for better throughput, simply specify "--mpi_num_processes=num_of_sockets [--mpi_num_processes_per_socket=1]". 
Note that the global batch size is mpi_num_processes * train_batch_size and sometimes learning rate needs to be adjusted for convergence. By default, the script uses square root learning rate scaling.
For fine-tuning tasks like BERT, state-of-the-art accuracy can be achieved via parallel training without synchronizing gradients between MPI workers. The "--mpi_workers_sync_gradients=[True/False]" controls whether the MPI workers sync gradients. By default it is set to "False" meaning the workers are training independently and the best performing training results will be picked in the end. To enable gradients synchronization, set the "--mpi_workers_sync_gradients" to `True` in the BERT-specific options.
The option `optimized_softmax=True` can also be set for better performance.


**Run BERT SQuAD Training on Multiple Sockets**

Create the output directory `large` to store the output results. 

```bash
mkdir large
```

Console in:

```bash
python launch_benchmark.py \
    --model-name=bert_large \
    --precision=bfloat16 \
    --mode=training \
    --framework=tensorflow \
    --batch-size=24 \
    --mpi_num_processes=4 \
    --docker-image intel/intel-optimized-tensorflow:latest \
    --volume $BERT_LARGE_DIR:$BERT_LARGE_DIR \
    --volume $SQUAD_DIR:$SQUAD_DIR \
    -- train_option=SQuAD \
       vocab_file=$BERT_LARGE_DIR/vocab.txt \
       config_file=$BERT_LARGE_DIR/bert_config.json \
       init_checkpoint=$BERT_LARGE_DIR/bert_model.ckpt \
       do_train=True \
       train_file=$SQUAD_DIR/train-v1.1.json \
       do_predict=True \
       predict_file=$SQUAD_DIR/dev-v1.1.json \
       learning_rate=3e-5 \
       num_train_epochs=2 \
       max_seq_length=384 \
       doc_stride=128 \
       output_dir=./large \
       optimized_softmax=True \
       experimental_gelu=False \
       do_lower_case=True
```

Note : For BERT-specific options, add a space after `--`.

If training was executed using the parameters above, the dev set predictions will have been saved to a file called `predictions.json` in the output directory `large`.

```bash
python $SQUAD_DIR/evaluate-v1.1.py $SQUAD_DIR/dev-v1.1.json ./large/predictions.json
```

An execution with these parameters produces results in line with below scores:

```
BF16: {"exact_match": 86.77388836329234, "f1": 92.98642358746287}
FP32: {"exact_match": 86.72658467360453, "f1": 92.98046893150796}
FP16: {"exact_match": 87.21854304635761, "f1": 93.35325056750564}
```

Please refer to google docs for SQuAD specific arguments.

**Run BERT Classifier Training on Multiple Sockets**

Download [The General Language Understanding Evaluation (GLUE) data](https://gluebenchmark.com/tasks) by running [this download script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e), and export the environment variable `GLUE_DIR`.
This example code fine-tunes BERT-Base on the Microsoft Research Paraphrase Corpus (MRPC) corpus, which only contains 3,600 examples.

```bash
wget https://gist.githubusercontent.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e/raw/17b8dd0d724281ed7c3b2aeeda662b92809aadd5/download_glue_data.py
pip install requests # required dependency 
python download_glue_data.py
export GLUE_DIR=$PWD/glue_data

export BERT_BASE_DIR=/home/<user>/wwm_uncased_L-12_H-768_A-16
```

Console in:

```bash
python launch_benchmark.py \
    --model-name=bert_large \
    --precision=bfloat16 \
    --mode=training \
    --framework=tensorflow \
    --batch-size=32 \
    --mpi_num_processes=4 \
    --docker-image intel/intel-optimized-tensorflow:latest \
    --volume $BERT_LARGE_DIR:$BERT_LARGE_DIR \
    --volume $GLUE_DIR:$GLUE_DIR \
    -- train-option=Classifier \
       task-name=MRPC \
       do-train=true \
       do-eval=true \
       data-dir=$GLUE_DIR/MRPC \
       vocab-file=$BERT_BASE_DIR/vocab.txt \
       config-file=$BERT_BASE_DIR/bert_config.json \
       init-checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
       max-seq-length=128 \
       learning-rate=2e-5 \
       num-train-epochs=30 \
       optimized_softmax=True \
       experimental_gelu=False \
       do-lower-case=True

```
The results files will be written to the
`models/benchmarks/common/tensorflow/logs` directory, unless another
output directory is specified by the `--output-dir` arg.

Note : For BERT-specific options, add a `space` after `--`.

The resulting output should be similar to the following:

```bash
***** Eval results *****
  eval_accuracy = 0.845588
  eval_loss = 0.505248
  global_step = 343
  loss = 0.505248
```

\*Other names and brands may be claimed as the property of others.
