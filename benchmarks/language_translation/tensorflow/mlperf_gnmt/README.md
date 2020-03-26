# MLPerf GNMT

## FP32 Inference Instruction

1. Prerequisites
* [tensorflow] 2.0.0rc0 or newer

2. Download GNMT benchmarking data.
```
wget https://zenodo.org/record/2531868/files/gnmt_inference_data.zip
```

3. Run inference using command from <REPO>/benchmarks dir
```
python launch_benchmark.py --model-name mlperf_gnmt --framework tensorflow --precision fp32 --mode inference --batch-size <some_number> --data-location <data_dir> --in-graph <frozen_graph_file> --accuracy_only
```
