#!/bin/bash

#Get the info of number of sockets,cores per socket, threads per core.
sockets=$(lscpu | grep "Socket(s)" | cut -d':' -f2 | xargs)
cores_per_socket=$(lscpu | grep "Core(s) per socket" | cut -d':' -f2 | xargs)
threads_per_core=$(lscpu | grep "Thread(s) per core" | cut -d':' -f2 | xargs)

for ARGUMENT in "$@"
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)  
    case "$KEY" in
            framework)              FRAMEWORK=${VALUE} ;;
            precision)              PRECISION=${VALUE} ;;
            mode)                   MODE=${VALUE} ;;
            model_name)             MODEL_NAME=${VALUE} ;;
            data_location)          DATA_LOCATION=${VALUE} ;;
            checkpoint)             CHECKPOINT=${VALUE} ;;
            in_graph)               IN_GRAPH=${VALUE} ;;
            output_dir)             OUTPUT_DIR=${VALUE} ;;
            infer_option)           INFER_OPTION=${VALUE} ;;
            batch_size)             BATCH_SIZE=${VALUE} ;;
            steps)                  STEPS=${VALUE} ;;
            *)   
    esac    
done

benchmark_only_arg="--benchmark-only"
RUN_SCRIPT_PATH="launch_benchmark.py"

CMD="numactl --cpunodebind=all --interleave=all python ${RUN_SCRIPT_PATH} \
--framework=${FRAMEWORK} \
--model-name=${MODEL_NAME} \
--precision=${PRECISION} \
--mode=${MODE} \
--batch-size=${BATCH_SIZE} \
--output-dir=${OUTPUT_DIR} \
--in-graph=${IN_GRAPH} \
${benchmark_only_arg} "

#Inter and Intra Threads are set after doing few experiments based on below link.
#https://software.intel.com/content/www/us/en/develop/articles/guide-to-tensorflow-runtime-optimizations-for-cpu.html#inpage-nav-1-undefined
if [ ${MODEL_NAME} == "bert_large" ]; then
  INTRA_THREADS=$cores_per_socket
  INTER_THREADS=$(($sockets * $cores_per_socket))
  CMD="${CMD} --infer_option=${INFER_OPTION}"
  CMD="${CMD} --checkpoint=${CHECKPOINT}"
  CMD="${CMD} --data-location=${DATA_LOCATION}"
fi

if [ ${MODEL_NAME} == "inceptionv3" ]; then
  INTRA_THREADS=$cores_per_socket
  INTER_THREADS=$(($sockets * $cores_per_socket))
  CMD="${CMD} --steps=${STEPS}"
fi

#For Wide-deep we will be using more number of threads than the number of physical cores available
#https://askubuntu.com/questions/668538/cores-vs-threads-how-many-threads-should-i-run-on-this-machine
if [ ${MODEL_NAME} == "wide_deep_large_ds" ]; then
  INTRA_THREADS=$(($sockets * $cores_per_socket * $threads_per_core))
  INTER_THREADS=$(($sockets * $cores_per_socket * $threads_per_core))
  CMD="${CMD} --data-location=${DATA_LOCATION}"
fi

CMD="${CMD} --num-intra-threads=${INTRA_THREADS} --num-inter-threads=${INTER_THREADS} --socket-id=0"
echo $CMD
$CMD

