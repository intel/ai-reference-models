
HOME=`pwd`
#echo $HOME

if [ $# -ne 3 ]; 
then 
	echo "illegal number of parameters"
	echo "$0 dataset model_file [fp32|int8]"
	exit 1
fi

dataset=$1
model=$2
precision=$3

rm -rf *.log

cd $HOME/../../../../benchmarks

python launch_benchmark.py \
    --in-graph $HOME/$model \
    --model-name resnet50 \
    --framework tensorflow \
    --precision $precision \
    --mode inference \
    --output-dir log \
    --batch-size 100 \
    --socket-id 0 \
    --data-location $HOME/$dataset | grep Throughput | tee $HOME/${precision}_throughput.txt
echo "save to ${precision}_throughput.txt"

python launch_benchmark.py \
    --in-graph $HOME/$model \
    --model-name resnet50 \
    --framework tensorflow \
    --precision $precision \
    --mode inference \
    --output-dir log \
    --batch-size 1 \
    --socket-id 0 \
    --data-location $HOME/$dataset | grep Average | tee $HOME/${precision}_latency.txt
echo "save to ${precision}_latency.txt"

python launch_benchmark.py \
    --in-graph $HOME/$model \
    --model-name resnet50 \
    --framework tensorflow \
    --precision $precision \
    --mode inference \
    --output-dir log \
    --accuracy-only \
    --batch-size 1000 \
    --socket-id 0 \
    --data-location $HOME/$dataset | grep accuracy | tee > $HOME/${precision}_accuracy.txt
echo "save to ${precision}_accuracy.txt"


