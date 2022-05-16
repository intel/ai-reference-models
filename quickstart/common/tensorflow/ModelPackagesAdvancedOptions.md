# Advanced Options for Model Packages and Containers

## Advanced model configuration

Model packages include example scripts that provide an easy way to get started
with various use cases such as running batch inference, getting accuracy metrics,
or training a model. These script typically have options to configure a path for
the dataset and an output directory. All quickstart scripts include 
a --dry-run option which will just echo to the terminal the call to launch_benchmark.py 
including its arguments. The --dry-run argument will be passed down to launch_benchmark.py
which will also echo the call to the model including its arguments. In general, quickstart scripts 
allow additional arguments to be passed to launch_benchmark.py and/or the underlying model script.
For example, quickstart scripts can be called with '--verbose' which will be passed to 
launch_benchmark.py or '--verbose -- steps=10' which will pass the verbose option to 
launch_benchmark.py and steps=10 to the underlying model script. Calling a 
quickstart script with '--help' will return all launch_benchmark.py options. 
Note: __replacing__ existing arguments or __deleting__ arguments within the quickstart script
isn't possible by calling the quickstart quick with additional arguments. For these types of cases
the user can add --dry-run and copy the echo'd result, making the needed edits. See the 
[launch benchmarks documentation](LaunchBenchmark.md) for information on the available arguments. 

Below are instructions for calling the `launch_benchmark.py` script with
bare metal and docker.

### Bare metal

Download and untar the model package, then call the `launch_benchmarks.py`
script in the benchmarks folder. The example below is showing how to do
this with the ResNet50 FP32 inference package, but the same process
can be used with other model packages.

```
DATASET_DIR=<path to the preprocessed imagenet dataset>
OUTPUT_DIR=<directory where log files will be written>

wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/resnet50-fp32-inference.tar.gz
tar -xzf resnet50_fp32_inference.tar.gz

cd resnet50_fp32_inference/benchmarks

python launch_benchmark.py \
  --model-name=resnet50 \
  --precision=fp32 \
  --mode=inference \
  --framework tensorflow \
  --in-graph ../resnet50_fp32_pretrained_model.pb \
  --data-location=${DATASET_DIR} \
  --output-dir ${OUTPUT_DIR} \
  --batch-size=128 \
  --socket-id 0
```

### Docker

Running the `launch_benchmarks.py` command in docker is similar to running
the quickstart scripts, where volume mounts and environment variables are set to
for the dataset and output directories. However, instead of having the
container execute the quickstart script, have it call
`python benchmarks/launch_benchmark.py` with your desired arguments. The
snippet below is showing how to do this using the ResNet50 FP32 inference
container, but the same process can be followed for other model usages by
swapping in the appropriate docker image/tag for your model.

```
DATASET_DIR=<path to the preprocessed imagenet dataset>
OUTPUT_DIR=<directory where log files will be written>

docker run \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  --privileged --init -t \
  intel/image-recognition:tf-latest-resnet50-fp32-inference \
  python benchmarks/launch_benchmark.py \
    --model-name resnet50 \
    --precision fp32 \
    --mode inference \
    --batch-size=128 \
    --socket-id 0 \
    --framework tensorflow \
    --output-dir {OUTPUT_DIR} \
    --in-graph resnet50_fp32_pretrained_model.pb \
    --data-location ${DATASET_DIR}
```

#### Docker CPUSET Settings

We recommend pinning a TensorFlow model container to a single NUMA node. Use the Docker run options `--cpuset-cpus` and `--cpuset-mems`. 
The output of the `lscpu` command will show the CPU IDs associated with each NUMA node.

***--cpuset-cpus/--cpuset-mems*** to confine the container to use a single NUMA node or the specified CPUs within a NUMA node.
For the equivalent of `numactl --cpunodebind=0 --membind=0`, use `docker run ... --cpuset-cpus="<cpus_in_numa_node_0>" --cpuset-mems=0 ...` for NUMA node 0.

An example for running `python benchmarks/launch_benchmark.py` using ResNet50 FP32 inference container, that is pinned to CPUs `0-27` in NUMA node 0:
<pre>
DATASET_DIR=&lt;path to the preprocessed imagenet dataset&gt;
OUTPUT_DIR=&lt;directory where log files will be written&gt;

docker run \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  <mark><b>--cpuset-cpus="0-27" --cpuset-mems="0"</b></mark> \
  --privileged --init -t \
  intel/image-recognition:tf-latest-resnet50-fp32-inference \
  python benchmarks/launch_benchmark.py \
    --model-name resnet50 \
    --precision fp32 \
    --mode inference \
    --batch-size=128 \
    --framework tensorflow \
    --output-dir {OUTPUT_DIR} \
    --in-graph resnet50_fp32_pretrained_model.pb \
    --data-location ${DATASET_DIR}
</pre>

If a cpuset is specified along with `--numa-cores-per-instance`, the cores
used for each instance will be limited to those specified as part of the cpuset.
Also, note that since `--numa-cores-per-instance` uses `numactl`, it needs to
be run with `--privilege`.

<pre>
$MODEL_ZOO_DIR=&lt;path to the model zoo directory&gt;
DATASET_DIR=&lt;path to the preprocessed imagenet dataset&gt;
OUTPUT_DIR=&lt;directory where log files will be written&gt;

docker run --rm --privileged --init \
    --volume $PRETRAINED_MODEL:$PRETRAINED_MODEL \
    --volume $MODEL_ZOO_DIR:$MODEL_ZOO_DIR \
    --volume $OUTPUT_DIR:$OUTPUT_DIR \
    --env http_proxy=$http_proxy \
    --env https_proxy=$https_proxy \
    --env PRETRAINED_MODEL=$PRETRAINED_MODEL \
    --env OUTPUT_DIR=$OUTPUT_DIR \
    -w $MODEL_ZOO_DIR \
    <mark><b>--cpuset-cpus "0-7,28-35"</b></mark> \
    -it intel/intel-optimized-tensorflow:latest \
    python benchmarks/launch_benchmark.py \
    --in-graph ${PRETRAINED_MODEL} \
    --model-name resnet50v1_5 \
    --framework tensorflow \
    --precision bfloat16 \
    --mode inference \
    --batch-size=1 \
    --output-dir ${OUTPUT_DIR} \
    --benchmark-only \
    <mark><b>--numa-cores-per-instance 4</b></mark>

# The command above ends up running the following instances:
# OMP_NUM_THREADS=4 numactl --localalloc --physcpubind=0,1,2,3 python eval_image_classifier_inference.py --input-graph=resnet50_v1_5_bfloat16.pb --num-inter-threads=1 --num-intra-threads=4 --num-cores=28 --batch-size=1 --warmup-steps=10 --steps=50 --data-num-inter-threads=1 --data-num-intra-threads=4 >> resnet50v1_5_bfloat16_inference_bs1_cores4_instance0.log 2>&1 & \
# OMP_NUM_THREADS=4 numactl --localalloc --physcpubind=4,5,6,7 python eval_image_classifier_inference.py --input-graph=resnet50_v1_5_bfloat16.pb --num-inter-threads=1 --num-intra-threads=4 --num-cores=28 --batch-size=1 --warmup-steps=10 --steps=50 --data-num-inter-threads=1 --data-num-intra-threads=4 >> resnet50v1_5_bfloat16_inference_bs1_cores4_instance1.log 2>&1 & \
# OMP_NUM_THREADS=4 numactl --localalloc --physcpubind=28,29,30,31 python eval_image_classifier_inference.py --input-graph=resnet50_v1_5_bfloat16.pb --num-inter-threads=1 --num-intra-threads=4 --num-cores=28 --batch-size=1 --warmup-steps=10 --steps=50 --data-num-inter-threads=1 --data-num-intra-threads=4 >> resnet50v1_5_bfloat16_inference_bs1_cores4_instance2.log 2>&1 & \
# OMP_NUM_THREADS=4 numactl --localalloc --physcpubind=32,33,34,35 python eval_image_classifier_inference.py --input-graph=resnet50_v1_5_bfloat16.pb --num-inter-threads=1 --num-intra-threads=4 --num-cores=28 --batch-size=1 --warmup-steps=10 --steps=50 --data-num-inter-threads=1 --data-num-intra-threads=4 >> resnet50v1_5_bfloat16_inference_bs1_cores4_instance3.log 2>&1 & \
# wait
</pre>

## Mounting local model packages in docker

A download of the model package can be run in a docker container by mounting the
model package to a directory a use case category
container (such as the category containers for image recognition, object
detection, etc). The category containers include library installs and
dependencies needed to run the model, but they do not have the model
package. This method of mounting the model package in the category
container allows you to locally make edits to the model package, but still
run the model using docker.

The example below shows the process of downloading and untarring the
model package and then using docker to mount directories for the dataset,
output, and the model package, and running a quickstart script. The model
package is being mounted to the `/workspace` directory in this example.
Note that this example is using ResNet50 FP32 inference, but the same
process can be used with other models by downloading a different model
package and using the category container that is approriate for that
model.

<pre>
DATASET_DIR=&lt;path to the preprocessed imagenet dataset&gt;
OUTPUT_DIR=&lt;directory where log files will be written&gt;

wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/resnet50-fp32-inference.tar.gz
tar -xzf resnet50_fp32_inference.tar.gz

docker run \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  <mark><b>--volume resnet50_fp32_inference:/workspace</b></mark> \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -w /workspace -t \
  <mark><b>intel/image-recognition:tf-latest-resnet50-fp32-inference</b></mark> \
  /bin/bash /workspace/quickstart/&lt;script name&gt;.sh
</pre>

## Running as the host user within the docker container

Often running as root within the container is not desired since any output generated by the model 
that is mounted on the host filesystem will be owned by root. This, in turn, requires the user to 
have sudo privileges on the host in order to remove, move or edit these files. By passing in the 
following env variables shown below, the user can run the model with their uid:gid permissions being passed into the container.

<pre>
DATASET_DIR=&lt;path to the preprocessed imagenet dataset&gt;
OUTPUT_DIR=&lt;directory where log files will be written&gt;

wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v2_7_0/resnet50-fp32-inference.tar.gz
tar -xzf resnet50_fp32_inference.tar.gz

docker run \
  <mark><b>--env USER_NAME=$(id -nu)</b></mark> \
  <mark><b>--env USER_ID=$(id -u)</b></mark> \
  <mark><b>--env GROUP_NAME=$(id -ng)</b></mark> \
  <mark><b>--env GROUP_ID=$(id -g)</b></mark> \
  --env DATASET_DIR=${DATASET_DIR} \
  --env OUTPUT_DIR=${OUTPUT_DIR} \
  --env http_proxy=${http_proxy} \
  --env https_proxy=${https_proxy} \
  <mark><b>--volume resnet50_fp32_inference:/workspace</b></mark> \
  --volume ${DATASET_DIR}:${DATASET_DIR} \
  --volume ${OUTPUT_DIR}:${OUTPUT_DIR} \
  --privileged --init -w /workspace -t \
  <mark><b>intel/image-recognition:tf-latest-resnet50-fp32-inference</b></mark> \
  /bin/bash /workspace/quickstart/&lt;script name&gt;.sh
</pre>
