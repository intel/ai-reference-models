# ResNet50 (v1.5)

This document has instructions for how to run ResNet50 (v1.5) for the
following precisions:
* [Int8 inference](#int8-inference-instructions)
* [FP32 inference](#fp32-inference-instructions)
* [BFloat16 inference](#bfloat16-inference-instructions)
* [FP32 training](#fp32-training-instructions)
* [BFloat16 training](#bfloat16-training-instructions)

Original ResNet model has multiple versions which have shown better accuracy
and/or batch inference and training performance. As mentioned in TensorFlow's [official ResNet
model page](https://github.com/tensorflow/models/tree/master/official/resnet), 3 different
versions of the original ResNet model exists - ResNet50v1, ResNet50v1.5, and ResNet50v2.
As a side note, ResNet50v1.5 is also in MLPerf's [cloud inference benchmark for
image
classification](https://github.com/mlperf/inference/tree/master/cloud/image_classification)
and [training benchmark](https://github.com/mlperf/training).

## Int8 Inference Instructions

1. Download the full ImageNet dataset and convert to the TF records format.

* Clone the tensorflow/models repository as `tensorflow-models`. This is to avoid conflict with Intel's `models` repo:
```
$ git clone https://github.com/tensorflow/models.git tensorflow-models
```
The TensorFlow models repo provides
[scripts and instructions](https://github.com/tensorflow/models/tree/master/research/slim#an-automated-script-for-processing-imagenet-data)
to download, process and convert the ImageNet dataset to the TF records format.

* The ImageNet dataset directory location is only required to calculate the model accuracy.

2. Download the pre-trained model.
```
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/resnet50v1_5_int8_pretrained_model.pb
```

3. Clone the
[intelai/models](https://github.com/intelai/models)
repository
```
$ git clone https://github.com/IntelAI/models.git
```

4. Run the inference script `launch_benchmark.py` with the appropriate parameters to evaluate the model performance and/or calculate the accuracy.
The optimized ResNet50v1.5 model files are attached to the [intelai/models](https://github.com/intelai/models) repo and
located at `models/models/image_recognition/tensorflow/resnet50v1_5/`.

    The docker image (`intel/intel-optimized-tensorflow:2.1.0`)
    used in the commands above were built using
    [TensorFlow](https://github.com/tensorflow/tensorflow.git) master for TensorFlow
    version 2.1.0.

* Calculate the model accuracy, the required parameters parameters include: the `ImageNet` dataset location (from step 1),
the pre-trained `resnet50v1_5_int8_pretrained_model.pb` input graph file (from step 2), and the `--accuracy-only` flag.
```
$ cd /home/<user>/models/benchmarks

$ python launch_benchmark.py \
    --data-location /home/<user>/dataset/FullImageNetData_directory \
    --in-graph resnet50v1_5_int8_pretrained_model.pb \
    --model-name resnet50v1_5 \
    --framework tensorflow \
    --precision int8 \
    --mode inference \
    --batch-size=100 \
    --accuracy-only \
    --docker-image intel/intel-optimized-tensorflow:2.1.0
```
The log file is saved to the value of `--output-dir`.

The tail of the log output when the benchmarking completes should look
something like this:
```
Iteration time: ... ms
Processed 49700 images. (Top1 accuracy, Top5 accuracy) = (0.7622, 0.9296)
Iteration time: ... ms
Processed 49800 images. (Top1 accuracy, Top5 accuracy) = (0.7621, 0.9295)
Iteration time: ... ms
Processed 49900 images. (Top1 accuracy, Top5 accuracy) = (0.7622, 0.9296)
Iteration time: ... ms
Processed 50000 images. (Top1 accuracy, Top5 accuracy) = (0.7623, 0.9296)
Ran inference with batch size 100
Log location outside container: {--output-dir value}/benchmark_resnet50_inference_int8_{timestamp}.log
```

* Evaluate the model performance: If just evaluate performance for dummy data, the `--data-location` is not needed.
Otherwise `--data-location` argument needs to be specified:
Calculate the batch inference performance `images/sec`, the required parameters to run the inference script would include:
the pre-trained `resnet50v1_5_int8_pretrained_model.pb` input graph file (from step
2), and the `--benchmark-only` flag. It is
optional to specify the number of `warmup_steps` and `steps` as extra
args, as shown in the command below. If these values are not specified,
the script will default to use `warmup_steps=10` and `steps=50`.

```
$ cd /home/<user>/models/benchmarks

$ python launch_benchmark.py \
    --in-graph resnet50v1_5_int8_pretrained_model.pb \
    --model-name resnet50v1_5 \
    --framework tensorflow \
    --precision int8 \
    --mode inference \
    --batch-size=128 \
    --benchmark-only \
    --docker-image intel/intel-optimized-tensorflow:2.1.0
    -- warmup_steps=50 steps=500
```
The tail of the log output when the benchmarking completes should look
something like this:
```
...
Iteration 490: ... sec
Iteration 500: ... sec
Average time: ... sec
Batch size = 128
Throughput: ... images/sec
Ran inference with batch size 128
Log location outside container: {--output-dir value}/benchmark_resnet50_inference_int8_{timestamp}.log
```

Note that the `--verbose` or `--output-dir` flag can be added to any of the above commands
to get additional debug output or change the default output location.

## FP32 Inference Instructions

1. Download the pre-trained model.

If you would like to get a pre-trained model for ResNet50v1.5,
```
$ wget https://zenodo.org/record/2535873/files/resnet50_v1.pb
```

2. Clone the [intelai/models](https://github.com/intelai/models) repository
```
$ git clone https://github.com/IntelAI/models.git
```

3. If running resnet50 for accuracy, the ImageNet dataset will be
required (if running the model for batch or online inference, then dummy
data will be used).

The TensorFlow models repo provides
[scripts and instructions](https://github.com/tensorflow/models/tree/master/research/slim#an-automated-script-for-processing-imagenet-data)
to download, process, and convert the ImageNet dataset to the TF records format.

4. Run the inference script `launch_benchmark.py` with the appropriate parameters to evaluate the model performance.
The optimized ResNet50v1.5 model files are attached to the [intelai/models](https://github.com/intelai/models) repo and
located at `models/models/image_recognition/tensorflow/resnet50v1_5/`.
If benchmarking uses dummy data for inference, `--data-location` flag is not required. Otherwise,
`--data-location` needs to point to point to ImageNet dataset location.

* To measure online inference, set `--batch-size=1` and run the model script as shown:
```
$ cd /home/<user>/models/benchmarks

$ python launch_benchmark.py \
    --in-graph resnet50_v1.pb \
    --model-name resnet50v1_5 \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --batch-size=1 \
    --socket-id=0 \
    --docker-image intel/intel-optimized-tensorflow:2.1.0
```

The log file is saved to the value of `--output-dir`.

The tail of the log output when the script completes should look
something like this:
```
Inference with dummy data.
Iteration 1: 2.761204 sec
Iteration 2: 0.011155 sec
Iteration 3: 0.009289 sec
...
Iteration 48: 0.009315 sec
Iteration 49: 0.009343 sec
Iteration 50: 0.009278 sec
Average time: 0.009481 sec
Batch size = 1
Latency: 9.481 ms
Throughput: 105.470 images/sec
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
Ran inference with batch size 1
Log location outside container: {--output-dir value}/benchmark_resnet50_inference_fp32_{timestamp}.log
```

* To measure batch inference, set `--batch-size=128` and run the model script as shown:
```
$ cd /home/<user>/models/benchmarks

$ python launch_benchmark.py \
    --in-graph resnet50_v1.pb \
    --model-name resnet50v1_5 \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --batch-size=128 \
    --socket-id=0 \
    --docker-image intel/intel-optimized-tensorflow:2.1.0
```

The log file is saved to the value of `--output-dir`.

The tail of the log output when the script completes should look
something like this:
```
Inference with dummy data.
Iteration 1: 3.013918 sec
Iteration 2: 0.543498 sec
Iteration 3: 0.536187 sec
Iteration 4: 0.532568 sec
...
Iteration 46: 0.532444 sec
Iteration 47: 0.535652 sec
Iteration 48: 0.532158 sec
Iteration 49: 0.538117 sec
Iteration 50: 0.532411 sec
Average time: 0.534427 sec
Batch size = 128
Throughput: 239.509 images/sec
Ran inference with batch size 128
Log location outside container: {--output-dir value}/benchmark_resnet50_inference_fp32_{timestamp}.log
```

* To measure the model accuracy, use the `--accuracy-only` flag and pass
the ImageNet dataset directory from step 3 as the `--data-location`:
```
$ cd /home/<user>/models/benchmarks

$ python launch_benchmark.py \
    --in-graph resnet50_v1.pb \
    --model-name resnet50v1_5 \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --accuracy-only \
    --batch-size 100 \
    --socket-id=0 \
    --data-location /home/<user>/dataset/ImageNetData_directory \
    --docker-image intel/intel-optimized-tensorflow:2.1.0
```

The log file is saved to the value of `--output-dir`.
The tail of the log output when the accuracy run completes should look
something like this:
```
...
Iteration time: 514.427 ms
Processed 50000 images. (Top1 accuracy, Top5 accuracy) = (0.7651, 0.9307)
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
Ran inference with batch size 100
Log location outside container: {--output-dir value}/benchmark_resnet50_inference_fp32_{timestamp}.log
```

* The `--output-results` flag can be used along with above performance
or accuracy test, in order to also output a file with the inference
results (file name, actual label, and the predicted label). The results
output can only be used with real data.

For example, the command below is the same as the accuracy test above,
except with the `--output-results` flag added:
```
$ cd /home/<user>/models/benchmarks

$ python launch_benchmark.py \
    --in-graph resnet50_v1.pb \
    --model-name resnet50v1_5 \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --accuracy-only \
    --output-results \
    --batch-size 100 \
    --socket-id=0 \
    --data-location /home/<user>/dataset/ImageNetData_directory \
    --docker-image intel/intel-optimized-tensorflow:2.1.0
```
The results file will be written to the
`models/benchmarks/common/tensorflow/logs` directory, unless another
output directory is specified by the `--output-dir` arg. Below is an
example of what the inference results file will look like:
```
filename,actual,prediction
ILSVRC2012_val_00033870.JPEG,592,592
ILSVRC2012_val_00045598.JPEG,258,258
ILSVRC2012_val_00047428.JPEG,736,736
ILSVRC2012_val_00003341.JPEG,344,344
ILSVRC2012_val_00037069.JPEG,192,192
ILSVRC2012_val_00029701.JPEG,440,440
ILSVRC2012_val_00016918.JPEG,286,737
ILSVRC2012_val_00015545.JPEG,5,5
ILSVRC2012_val_00016713.JPEG,274,274
ILSVRC2012_val_00014735.JPEG,31,31
...
```

Note that the `--verbose` or `--output-dir` flag can be added to any of the above commands
to get additional debug output or change the default output location.

## BFloat16 Inference Instructions
(Experimental)

1. Download the pre-trained model.

```
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6_1/resnet50_v1_5_bfloat16.pb
```

2. Clone the [intelai/models](https://github.com/intelai/models) repository
```
$ git clone https://github.com/IntelAI/models.git
```

3. If running resnet50 for accuracy, the ImageNet dataset will be
required (if running the model for batch or online inference, then dummy
data will be used).

The TensorFlow models repo provides
[scripts and instructions](https://github.com/tensorflow/models/tree/master/research/slim#an-automated-script-for-processing-imagenet-data)
to download, process, and convert the ImageNet dataset to the TF records format.

4. Run the inference script `launch_benchmark.py` with the appropriate parameters to evaluate the model performance.
The optimized ResNet50v1.5 model files are attached to the [intelai/models](https://github.com/intelai/models) repo and
located at `models/models/image_recognition/tensorflow/resnet50v1_5/`.
If benchmarking uses dummy data for inference, `--data-location` flag is not required. Otherwise,
`--data-location` needs to point to ImageNet dataset location.

* To measure online inference, set `--batch-size=1` and run the model script as shown:
```
$ cd /home/<user>/models/benchmarks

$ python launch_benchmark.py \
    --in-graph resnet50_v1_bfloat16.pb \
    --model-name resnet50v1_5 \
    --framework tensorflow \
    --precision bfloat16 \
    --mode inference \
    --batch-size=1 \
    --socket-id 0 \
    --docker-image=intel/intel-optimized-tensorflow:tensorflow-2.2-bf16-nightly
```

The log file is saved to the value of `--output-dir`.

The tail of the log output when the script completes should look
something like this:
```
Inference with dummy data.
Iteration 1: ... sec
Iteration 2: ... sec
Iteration 3: ... sec
...
Iteration 48: ... sec
Iteration 49: ... sec
Iteration 50: ... sec
Average time: ... sec
Batch size = 1
Latency: ... ms
Throughput: ... images/sec
Ran inference with batch size 1
Log location outside container: {--output-dir value}/benchmark_resnet50_inference_bfloat16_{timestamp}.log
```

* To measure batch inference, set `--batch-size=128` and run the model script as shown:
```
$ cd /home/<user>/models/benchmarks

$ python launch_benchmark.py \
    --in-graph resnet50_v1_bfloat16.pb \
    --model-name resnet50v1_5 \
    --framework tensorflow \
    --precision bfloat16 \
    --mode inference \
    --batch-size=128 \
    --socket-id 0 \
    --docker-image=intel/intel-optimized-tensorflow:tensorflow-2.2-bf16-nightly
```

The log file is saved to the value of `--output-dir`.

The tail of the log output when the script completes should look
something like this:
```
Inference with dummy data.
Iteration 1: ... sec
Iteration 2: ... sec
Iteration 3: ... sec
...
Iteration 47: ... sec
Iteration 48: ... sec
Iteration 49: ... sec
Iteration 50: ... sec
Average time: ... sec
Batch size = 128
Throughput: ... images/sec
Ran inference with batch size 128
Log location outside container: {--output-dir value}/benchmark_resnet50_inference_bfloat16_{timestamp}.log
```

* To measure the model accuracy, use the `--accuracy-only` flag and pass
the ImageNet dataset directory from step 3 as the `--data-location`:
```
$ cd /home/<user>/models/benchmarks

$ python launch_benchmark.py \
    --in-graph resnet50_v1_bfloat16.pb \
    --model-name resnet50v1_5 \
    --framework tensorflow \
    --precision bfloat16 \
    --mode inference \
    --accuracy-only \
    --batch-size 100 \
    --socket-id 0 \
    --data-location /home/<user>/dataset/ImageNetData_directory \
    --docker-image=intel/intel-optimized-tensorflow:tensorflow-2.2-bf16-nightly
```

The log file is saved to the value of `--output-dir`.
The tail of the log output when the accuracy run completes should look
something like this:
```
...
Iteration time: ... ms
Processed 50000 images. (Top1 accuracy, Top5 accuracy) = (0.7674, 0.9316)
Ran inference with batch size 100
Log location outside container: {--output-dir value}/benchmark_resnet50_inference_bfloat16_{timestamp}.log
```

* The `--output-results` flag can be used along with above performance
or accuracy test, in order to also output a file with the inference
results (file name, actual label, and the predicted label). The results
output can only be used with real data.

For example, the command below is the same as the accuracy test above,
except with the `--output-results` flag added:
```
$ cd /home/<user>/models/benchmarks

$ python launch_benchmark.py \
    --in-graph resnet50_v1_bfloat16.pb \
    --model-name resnet50v1_5 \
    --framework tensorflow \
    --precision bfloat16 \
    --mode inference \
    --accuracy-only \
    --output-results \
    --batch-size 100 \
    --socket-id 0 \
    --data-location /home/<user>/dataset/ImageNetData_directory \
    --docker-image=intel/intel-optimized-tensorflow:tensorflow-2.2-bf16-nightly
```
The results file will be written to the
`models/benchmarks/common/tensorflow/logs` directory, unless another
output directory is specified by the `--output-dir` arg. Below is an
example of what the inference results file will look like:
```
filename,actual,prediction
ILSVRC2012_val_00033870.JPEG,592,592
ILSVRC2012_val_00045598.JPEG,258,258
ILSVRC2012_val_00047428.JPEG,736,736
ILSVRC2012_val_00003341.JPEG,344,344
ILSVRC2012_val_00037069.JPEG,192,192
ILSVRC2012_val_00029701.JPEG,440,440
ILSVRC2012_val_00016918.JPEG,286,737
ILSVRC2012_val_00015545.JPEG,5,5
ILSVRC2012_val_00016713.JPEG,274,274
ILSVRC2012_val_00014735.JPEG,31,31
...
```

Note that the `--verbose` or `--output-dir` flag can be added to any of the above commands
to get additional debug output or change the default output location.

## FP32 Training Instructions

1. Download the full ImageNet dataset and convert to the TF records format.

* Clone the tensorflow/models repository as `tensorflow-models`. This is to avoid conflict with Intel's `models` repo:
```
$ git clone https://github.com/tensorflow/models.git tensorflow-models
```
The TensorFlow models repo provides
[scripts and instructions](https://github.com/tensorflow/models/tree/master/research/slim#an-automated-script-for-processing-imagenet-data)
to download, process and convert the ImageNet dataset to the TF records format.

2. Clone the
[intelai/models](https://github.com/intelai/models)
repository
```
$ git clone https://github.com/IntelAI/models.git
```

3. Run the following command to start ResNet50v1.5 FP32 training run.
```
$ python launch_benchmark.py \
         --model-name=resnet50v1_5 \
         --precision=fp32 \
         --mode=training \
         --framework tensorflow \
         --checkpoint <location_to_store_training_checkpoints> \
         --data-location=/home/<user>/dataset/ImageNetData_directory
```

This run will take considerable amount of time since it is running for
convergence (90 epochs).

If you want to do a trial run, add
```
-- steps=<number_of_training_steps>
```
argument to the command.

If you run the script for more than 100 steps, you should see training loss
decreasing like below:

```
I0816 basic_session_run_hooks.py:262] loss = 8.442491, step = 0
I0816 basic_session_run_hooks.py:260] loss = 8.373407, step = 100 (... sec)
...
```

## BFloat16 Training Instructions
(Experimental)

1. Download the full ImageNet dataset and convert to the TF records format.

* Clone the tensorflow/models repository:
```
$ git clone https://github.com/tensorflow/models.git
```
The TensorFlow models repo provides
[scripts and instructions](https://github.com/tensorflow/models/tree/master/research/slim#an-automated-script-for-processing-imagenet-data)
to download, process and convert the ImageNet dataset to the TF records format.

2. Clone the
[intelai/models](https://github.com/intelai/models)
repository
```
$ git clone https://github.com/IntelAI/models.git
```

3. Run the following command to start ResNet50v1.5 BFloat16 training run.
```
$ python launch_benchmark.py \
         --model-name=resnet50v1_5 \
         --precision=bfloat16 \
         --mode=training \
         --framework tensorflow \
         --checkpoint <location_to_store_training_checkpoints> \
         --data-location=/home/<user>/dataset/ImageNetData_directory \
         --docker-image=intel/intel-optimized-tensorflow:tensorflow-2.2-bf16-nightly
```

This run will take considerable amount of time since it is running for
convergence (90 epochs).

If you want to do a trial run, add
```
-- steps=<number_of_training_steps>
```
argument to the command.

If you run the script for more than 100 steps, you should see training loss
decreasing like below:

```
I0816 basic_session_run_hooks.py:262] loss = 8.442491, step = 0
I0816 basic_session_run_hooks.py:260] loss = 8.373407, step = 100 (... sec)
...
```
## Distributed Training Instructions
Training can be done in a distributed fashion. On a dual (or eight) socket system, one can create two (or eight) MPI processes (one socket each) to do the training. As an example, run the following command to start ResNet50v1.5 FP32 training run using 2 MPI processes.
```
$ python launch_benchmark.py \
         --model-name=resnet50v1_5 \
         --precision=bfloat16 \
         --mode=training \
         --framework tensorflow \
         --data-location=/home/<user>/dataset/ImageNetData_directory \
         --mpi_num_processes=2 \
         --docker-image=intel/intel-optimized-tensorflow:tensorflow-2.2-bf16-nightly
```
The above distributed training runs one MPI process per socket, to maximize performance, users can run more than one (commonly two) MPI processes per socket. The following command achieves launching 4 MPI processes over 2 sockets. Note that in this case we need to reduce the OMP_NUM_THREADS and intra_op_parallelism_threads by half (minus one or two for performance sometimes, e.g. half of 28 becomes 14, and we can use 12 for good performance).  This is controlled by "-a <half the amount of cores of per socket or less>". Batch size can remain the same for weak scaling or reduced by half as well for strong scaling.

```
$ python launch_benchmark.py \
         --model-name=resnet50v1_5 \
         --precision=bfloat16 \
         --mode=training \
         --framework tensorflow \
         --data-location=/home/<user>/dataset/ImageNetData_directory \
         --mpi_num_processes=4 \
         --mpi_num_processes_per_socket=2 \
         --docker-image=intel/intel-optimized-tensorflow:tensorflow-2.2-bf16-nightly \
         -a <half the amount of cores per socket or less>
```

Similarly, the following command achieves launching 2 MPI processes over 1 socket.

```
$ python launch_benchmark.py \
         --model-name=resnet50v1_5 \
         --precision=bfloat16 \
         --mode=training \
         --framework tensorflow \
         --data-location=/home/<user>/dataset/ImageNetData_directory \
         --mpi_num_processes=2 \
         --mpi_num_processes_per_socket=2 \
         --docker-image=intel/intel-optimized-tensorflow:tensorflow-2.2-bf16-nightly \
         -a <half the amount of cores per socket or less>
```

You can check output trained model accuracy by setting `--eval=True` in the command. After training is over, it automatically run inference and report accuracy results.

ResNet50 convergence with MLPerf target Top-1 accuracy of 75.9% can be achieved using the following set of instructions. The instructions have been tested on a 8-socket 28-core Xeon server.
```
$ cd models/image_recognition/tensorflow/resnet50v1_5/training/mlperf_resnet
$ RANDOM_SEED=`date +%s`

# Save the original PYTHONPATH
$ export ORIG_PYTHONPATH="${PYTHONPATH}"

# Register the model as a source root
$ export PYTHONPATH="$(pwd):${PYTHONPATH}"

# Add MLPerf source to PYTHONPATH
$ export PYTHONPATH="$(pwd)/../:$(pwd)/../../../../../common/tensorflow/:${PYTHONPATH}"

# Set path for saving checkpoints
$ MODEL_DIR=“PATH_TO_CHECKPOINT/resnet_imagenet_${RANDOM_SEED}"
  
$ export OMP_NUM_THREADS=4
$ export KMP_BLOCKTIME=1
  
$ mpirun --allow-run-as-root -n 32  --map-by ppr:4:socket:pe=7 python imagenet_main.py $RANDOM_SEED \
  --data_dir PATH_TO_DATASET/imagenet_dataset --model_dir $MODEL_DIR --train_epochs 10000 \
  --stop_threshold 0.759 --batch_size 128 --version 1 --resnet_size 50 --epochs_between_evals 4 --weight_decay 1e-4 \
  --inter_op_parallelism_threads 2 --intra_op_parallelism_threads 4 --fine_tune --use_bfloat16

# Restore the original PYTHONPATH
export PYTHONPATH="${ORIG_PYTHONPATH}"
```
--map-by ppr:4:socket:pe=7 starts 4 mpi processes per socket and binds each of them to 7 cores

--fine_tune is optional. Adding this param sets base learning rate to 0.1. The default base learning rate (when —fine-tune is not set) is 0.128. Convergence with mlperf target accuracy of 75.9% has been achieved using both 0.1 and 0.128.

Although --train_epochs is set to 1000, target accuracy of 75.9% can be reached in 84 epochs. --stop_threshold ensures that training will stop once target accuracy of 75.9% has been reached.

For bfloat16 training, pass the param --use_bfloat16 in mpirun command (as indicated above). For FP32, do not pass this param. Other than this difference, rest of instructions remain same for convergence with bfloat16 and FP32.

For running distributed training on a cluster, modify params passed to mpirun (add --host … , modify --map-by …), OMP_NUM_THREADS and model hyper parameters depending on hardware configuration of machines in the cluster
