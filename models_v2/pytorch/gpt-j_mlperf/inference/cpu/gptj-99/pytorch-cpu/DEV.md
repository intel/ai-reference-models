## Setup Instructions
!! This guide is for develop purposes only and is not officially supported.

### Download and Prepare Dataset
```
export CUR_DIR=$(pwd)
export WORKLOAD_DATA=${CUR_DIR}/data
mkdir -p ${WORKLOAD_DATA}
```

+ Download cnn-dailymail calibration set
```
python download-calibration-dataset.py --calibration-list-file calibration-list.txt --output-dir ${WORKLOAD_DATA}/calibration-data
```

+ Download cnn-dailymail validation set
```
python download-dataset.py --split validation --output-dir ${WORKLOAD_DATA}/validation-data
```

### Download and prepare model
+ Get finetuned checkpoint
```
export CHECKPOINT_DIR=${WORKLOAD_DATA}/gpt-j-checkpoint
wget https://cloud.mlcommons.org/index.php/s/QAZ2oM94MkFtbQx/download -O gpt-j-checkpoint.zip
unzip gpt-j-checkpoint.zip
mv gpt-j/checkpoint-final/ ${CHECKPOINT_DIR}
```

+ Get quantized model: Navigate to `../../../calibration/gptj-99/pytorch-cpu` and follow the README. Then return here to run benchmarks

### Run Benchmarks
+ Offline (Performance)
```
bash run_offline_int4.sh # for INT4
```

+ Offline (Accuracy)
```
bash run_offline_accuracy_int4.sh # for INT4
```

+ Server (Performance)
```
bash run_server_int4.sh # for INT4
```

+ Server (Accuracy)
```
bash run_server_accuracy_int4.sh # for INT4
```

## Docker Instructions
Using the dockerfile provided in `docker/Dockerfile` and `docker/build_gpt-j_int4_container.sh`, user can build and run the benchmarks following the instructions below

### Building the container

+ Since the Docker build step copies the `gpt-j` directory and its subdirectories, the `${WORKLOAD_DATA}` and `gpt-j-env` directories (which can be very large) have to be moved if they're present in the current folder. Skip this if not applicable
```
mv ${WORKLOAD_DATA} ../../gpt-j-data [Or your selected path]
mv gpt-j-env ../../gpt-j-env
```

+ Now build the docker image
```
cd docker
bash build_gpt-j_int4_container.sh
cd ..
```
+ Start the container
```
ln -s ../../gpt-j-data data # Create softlink to the moved workload data
source setup_env.sh
docker run --name intel_gptj --privileged -itd --net=host --ipc=host -v ${WORKLOAD_DATA}:/opt/workdir/code/gptj-99/pytorch-cpu/data mlperf_inference_gptj:3.1
docker exec -it intel_gptj bash
``` 
 ### Calibration
 Please follow "Download and prepare model" section.

 ### Start benchmarks

+ Offline (Performance)
```
bash run_offline_int4.sh
```

+ Offline (Accuracy)
```
bash run_offline_accuracy_int4.sh
```

+ Server (Performance)
```
bash run_server_int4.sh
```

+ Server (Accuracy)
```
bash run_server_accuracy_int4.sh
```
