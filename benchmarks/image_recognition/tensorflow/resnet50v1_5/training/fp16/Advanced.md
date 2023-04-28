<!--- 0. Title -->
<!-- This document is auto-generated using markdown fragments and the model-builder -->
<!-- To make changes to this doc, please change the fragments instead of modifying this doc directly -->
# ResNet50 v1.5 FP16 training - Advanced Instructions

<!-- 10. Description -->
This document has advanced instructions for running ResNet50 v1.5 FP16
training, which provides more control over the individual parameters that
are used. For more information on using [`/benchmarks/launch_benchmark.py`](/benchmarks/launch_benchmark.py),
see the [launch benchmark documentation](/docs/general/tensorflow/LaunchBenchmark.md).

Prior to using these instructions, please follow the setup instructions from
the model's [README](README.md) and/or the
[AI Kit documentation](/docs/general/tensorflow/AIKit.md) to get your environment
setup (if running on bare metal) and download the dataset, pretrained model, etc.
If you are using AI Kit, please exclude the `--docker-image` flag from the
commands below, since you will be running the the TensorFlow conda environment
instead of docker.

<!-- 55. Docker arg -->
Any of the `launch_benchmark.py` commands below can be run on bare metal by
removing the `--docker-image` arg. Ensure that you have all of the
[required prerequisites installed](README.md#run-the-model) in your environment
before running without the docker container.

If you are new to docker and are running into issues with the container,
see [this document](/docs/general/docker.md) for troubleshooting tips.

<!-- 50. Launch benchmark instructions -->
Once your environment is setup, navigate to the `benchmarks` directory of
the model zoo and set environment variables pointing to the directory for the
dataset, number of MPI processes, and an output directory where log files will 
be written.

```
# cd to the benchmarks directory in the model zoo
cd benchmarks

export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
export MPI_NUM_PROCESSES=<desired number of MPI processes (optional)>
```

ResNet50 v1.5 FP16 training can be run to test full training, 
single-epoch training, or a training demo. Use one of the following examples 
below, depending on your use case.

* For full training to convergence run the following command that uses the `DATASET_DIR`
  and an `--mpi_num_processes` argument which defaults to 1:

```
python launch_benchmark.py \
  --model-name=resnet50v1_5 \
  --precision=fp16 \
  --mode=training \
  --framework tensorflow \
  --checkpoint ${OUTPUT_DIR} \
  --data-location=${DATASET_DIR} \
  --output-dir ${OUTPUT_DIR} \
  --mpi_num_processes=${MPI_NUM_PROCESSES} \
  --docker-image intel/intel-optimized-tensorflow:latest
```

* For a single epoch of training, use the command below that uses the `DATASET_DIR`,
  a `-- steps=<number_of_training_steps>` argument, and an `--mpi_num_processes` argument which defaults to 1:

```
python launch_benchmark.py \
  --model-name=resnet50v1_5 \
  --precision=fp16 \
  --mode=training \
  --framework tensorflow \
  --batch-size 256 \
  --checkpoint ${OUTPUT_DIR} \
  --data-location=${DATASET_DIR} \
  --output-dir ${OUTPUT_DIR} \
  --mpi_num_processes=${MPI_NUM_PROCESSES} \
  --docker-image intel/intel-optimized-tensorflow:latest \
  -- steps=100 train_epochs=1 epochs_between_evals=1
```

* For an even shorter training demo (only 50 steps), use the command below that uses 
  the `DATASET_DIR`, a `-- steps=<number_of_training_steps>` argument, and an 
  `--mpi_num_processes` argument which defaults to 1:
  
```
python launch_benchmark.py \
  --model-name=resnet50v1_5 \
  --precision=fp16 \
  --mode=training \
  --framework tensorflow \
  --batch-size 16 \
  --checkpoint ${OUTPUT_DIR} \
  --data-location=${DATASET_DIR} \
  --mpi_num_processes=${MPI_NUM_PROCESSES} \
  --docker-image intel/intel-optimized-tensorflow:latest \
  -- steps=50 train_epochs=1 epochs_between_evals=1
```

If you run the script for more than 100 steps, you should see training loss
decreasing like below:

```
I0816 basic_session_run_hooks.py:262] loss = 8.442491, step = 0
I0816 basic_session_run_hooks.py:260] loss = 8.373407, step = 100 (... sec)
...
```

## Distributed Training
Training can be done in a distributed fashion. On a dual (or eight) socket system, 
one can create two (or eight) MPI processes (one socket each) to do the training. 
As an example, run the following command to start ResNet50v1.5 FP32 training run using 
2 MPI processes.
```
python launch_benchmark.py \
  --model-name resnet50v1_5 \
  --precision fp16 \
  --mode training \
  --framework tensorflow \
  --data-location ${DATASET_DIR} \
  --output-dir ${OUTPUT_DIR} \
  --mpi_num_processes 2 \
  --docker-image intel/intel-optimized-tensorflow:latest
```
The above distributed training runs one MPI process per socket, to maximize performance.
Users can run more than one (commonly two) MPI processes per socket. The following command 
achieves launching 4 MPI processes over 2 sockets. Note that in this case we need to reduce 
the OMP_NUM_THREADS and intra_op_parallelism_threads by half (minus one or two for performance 
sometimes, e.g. half of 28 becomes 14, and we can use 12 for good performance).
This is controlled by "-a <half the amount of cores of per socket or less>".
Batch size can remain the same for weak scaling or reduced by half as well for strong scaling.

```
python launch_benchmark.py \
  --model-name resnet50v1_5 \
  --precision fp16 \
  --mode training \
  --framework tensorflow \
  --data-location ${DATASET_DIR} \
  --output-dir ${OUTPUT_DIR} \
  --mpi_num_processes 4 \
  --mpi_num_processes_per_socket 2 \
  --docker-image intel/intel-optimized-tensorflow:latest \
  -a <half the amount of cores per socket or less>
```

Similarly, the following command achieves launching 2 MPI processes over 1 socket.

```
python launch_benchmark.py \
  --model-name resnet50v1_5 \
  --precision fp16 \
  --mode training \
  --framework tensorflow \
  --data-location ${DATASET_DIR} \
  --output-dir ${OUTPUT_DIR} \
  --mpi_num_processes 2 \
  --mpi_num_processes_per_socket 1 \
  --docker-image intel/intel-optimized-tensorflow:latest \
  -a <half the amount of cores per socket or less>
```

You can check output trained model accuracy by setting `--eval=True` in the command. 
After training is over, it automatically run inference and report accuracy results.

Finally, the following command runs MPI across multiple nodes on bare-metal, with 2 MPI processes 
per node. Each node must have passwordless ssh enabled for the user running the command below. 
All hosts should have these additional packages installed: (apt-get) openmpi-bin openmpi-common 
libopenmpi-dev, (pip) horovod==0.20.0

```
python launch_benchmark.py \
  --verbose \
  --model-name resnet50v1_5 \
  --precision fp16 \
  --mode training \
  --framework tensorflow \
  --noinstall \
  --checkpoint ${OUTPUT_DIR} \
  --output-dir ${OUTPUT_DIR} \
  --data-location ${DATASET_DIR} \
  --mpi_hostnames 'host1,host2' \
  --mpi_num_processes 4 2>&1
```
