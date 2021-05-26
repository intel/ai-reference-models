<!-- 50. Launch benchmark instructions -->
Once your environment is setup, navigate to the `benchmarks` directory of
the model zoo and set environment variables pointing to the directory for the
pretrained model, model repository, and an output directory where log
files will be written.

```
# cd to the benchmarks directory in the model zoo
cd benchmarks

export OUTPUT_DIR=<directory where log files will be written>
export PRETRAINED_MODEL=<path to the pretrained model>
export TF_UNET_DIR=<path to the TF UNet directory tf_unet>
```

<model name> <precision> <mode> can be run to test batch and online inference using the
following command:
```
python launch_benchmark.py \
  --model-name unet \
  --precision fp32 \
  --mode inference \
  --framework tensorflow \
  --benchmark-only \
  --batch-size 1 \
  --socket-id 0 \
  --checkpoint ${PRETRAINED_MODEL} \
  --model-source-dir ${TF_UNET_DIR} \
  --output-dir ${OUTPUT_DIR} \
  --docker-image <docker image> \
  -- checkpoint_name=model.ckpt
```

Below is an example of the log file tail:

```
Time spent per BATCH: ... ms
Total samples/sec: ... samples/s
Ran inference with batch size 1
Log location outside container: {--output-dir value}/benchmark_unet_inference_fp32_20190201_205601.log
```
