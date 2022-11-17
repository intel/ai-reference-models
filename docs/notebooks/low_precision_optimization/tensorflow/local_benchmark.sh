WORK_DIR=`pwd`

if [ $# -ne 3 ]; then 
    echo "illegal number of parameters"
    echo "$0 dataset model_file [fp32|int8]"
    exit 1
fi

DATASET=$1
MODEL=$2
PRECISION=$3

rm -rf *.log

cd ${WORK_DIR}/../../../../benchmarks

python launch_benchmark.py \
    --in-graph ${WORK_DIR}/${MODEL} \
    --model-name resnet50 \
    --framework tensorflow \
    --precision ${PRECISION} \
    --mode inference \
    --output-dir log \
    --batch-size 100 \
    --socket-id 0 \
    --data-location ${WORK_DIR}/${DATASET} | grep Throughput | tee ${WORK_DIR}/${PRECISION}_throughput.txt
echo "save to ${PRECISION}_throughput.txt"

python launch_benchmark.py \
    --in-graph ${WORK_DIR}/${MODEL} \
    --model-name resnet50 \
    --framework tensorflow \
    --precision ${PRECISION} \
    --mode inference \
    --output-dir log \
    --batch-size 1 \
    --socket-id 0 \
    --data-location ${WORK_DIR}/${DATASET} | grep Average | tee ${WORK_DIR}/${PRECISION}_latency.txt
echo "save to ${PRECISION}_latency.txt"

python launch_benchmark.py \
    --in-graph ${WORK_DIR}/${MODEL} \
    --model-name resnet50 \
    --framework tensorflow \
    --precision ${PRECISION} \
    --mode inference \
    --output-dir log \
    --accuracy-only \
    --batch-size 1000 \
    --socket-id 0 \
    --data-location ${WORK_DIR}/${DATASET} | grep accuracy | tee > ${WORK_DIR}/${PRECISION}_accuracy.txt
echo "save to ${PRECISION}_accuracy.txt"


