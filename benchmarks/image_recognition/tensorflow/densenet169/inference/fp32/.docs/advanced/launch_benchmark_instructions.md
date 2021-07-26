<!-- 50. Launch benchmark instructions -->
Once your environment is setup, navigate to the `benchmarks` directory of
the model zoo and set environment variables pointing to the directory for the
dataset, pretrained model frozen graph, and an output directory where log
files will be written.

```
# cd to the benchmarks directory in the model zoo
cd benchmarks

export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
export PRETRAINED_MODEL=<path to the frozen graph that you downloaded>
```

Run throughput benchmarking with batch size 100 using the following command:
```
python launch_benchmark.py \
    --model-name densenet169 \
    --precision fp32 \
    --mode inference \
    --framework tensorflow \
    --benchmark-only \
    --batch-size 100 \
    --socket-id 0 \
    --in-graph ${PRETRAINED_MODEL} \
    --docker-image intel/intel-optimized-tensorflow:latest \
    --output-dir ${OUTPUT_DIR} \
    -- input_height=224 input_width=224 warmup_steps=20 steps=100 \
    input_layer="input" output_layer="densenet169/predictions/Reshape_1"
```

Run latency benchmarking with batch size 1 using the following command:
```
python launch_benchmark.py \
    --model-name densenet169 \
    --precision fp32 \
    --mode inference \
    --framework tensorflow \
    --benchmark-only \
    --batch-size 1 \
    --socket-id 0 \
    --in-graph ${PRETRAINED_MODEL} \
    --docker-image intel/intel-optimized-tensorflow:latest \
    --output-dir ${OUTPUT_DIR} \
    -- input_height=224 input_width=224 warmup_steps=20 steps=100 \
    input_layer="input" output_layer="densenet169/predictions/Reshape_1"
```

Run an accuracy test with the following command:
```
python launch_benchmark.py \
    --model-name densenet169 \
    --precision fp32 \
    --mode inference \
    --framework tensorflow \
    --accuracy-only \
    --batch-size 100 \
    --socket-id 0 \
    --in-graph ${PRETRAINED_MODEL} \
    --docker-image intel/intel-optimized-tensorflow:latest \
    --data-location ${DATASET_DIR} \
    --output-dir ${OUTPUT_DIR} \
    -- input_height=224 input_width=224 \
    input_layer="input" output_layer="densenet169/predictions/Reshape_1"
```

Output files and logs are saved to the `${OUTPUT_DIR}` directory. Below are
examples of what the tail of your log file should look like for the different configs.

Example log tail when running for batch inference:
```
steps = 80, 159.83471377 images/sec
       Latency: 625.646317005 ms
steps = 90, 159.852789241 images/sec
       Latency: 625.57557159 ms
steps = 100, 159.853966416 images/sec
       Latency: 625.570964813 ms
Ran inference with batch size 100
Log location outside container: ${OUTPUT_DIR}/benchmark_densenet169_inference_fp32_20190412_023940.log
```

Example log tail when running for online inference:
```
steps = 80, 34.9948442873 images/sec
       Latency: 28.5756379366 ms
steps = 90, 34.9644341907 images/sec
       Latency: 28.6004914178 ms
steps = 100, 34.9655988121 images/sec
       Latency: 28.5995388031 ms
Ran inference with batch size 1
Log location outside container: ${OUTPUT_DIR}/benchmark_densenet169_inference_fp32_20190412_024505.log
```

Example log tail when running for accuracy:
```
Iteration time: 581.6446 ms
0.757505030181
Iteration time: 581.5755 ms
0.757489959839
Iteration time: 581.5709 ms
0.75749498998
Iteration time: 581.1705 ms
0.75748
Ran inference with batch size 100
Log location outside container: ${OUTPUT_DIR}/benchmark_densenet169_inference_fp32_20190412_021545.log
```
