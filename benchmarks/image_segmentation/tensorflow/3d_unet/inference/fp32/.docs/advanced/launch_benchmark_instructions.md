<!-- 50. Launch benchmark instructions -->
Once your environment is setup, navigate to the `benchmarks` directory of
the model zoo and set environment variables pointing to the directory for the
dataset, pretrained model, and an output directory where log
files will be written.

```
# cd to the benchmarks directory in the model zoo
cd benchmarks

export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
export PRETRAINED_MODEL=<path to the pretrained model>
```

<model name> <precision> <mode> can be run with a batch size of 1:
```
python launch_benchmark.py \
  --precision fp32 \
  --model-name 3d_unet \
  --mode inference \
  --framework tensorflow \
  --docker-image <docker image> \
  --in-graph $PRETRAINED_MODEL \
  --data-location ${DATASET_DIR} \
  --batch-size 1 \
  --socket-id 0 \
  --output-dir ${OUTPUT_DIR} 
```

Below is an example tail of the log file:

```
Loading pre-trained model
Time spent per BATCH: ... ms
Total samples/sec: ... samples/s
Ran inference with batch size 1
Log location outside container: {--output-dir value}/benchmark_3d_unet_inference_fp32_20190116_234659.log
```
