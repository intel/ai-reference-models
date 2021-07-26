<!-- 50. Launch benchmark instructions -->
Once your environment is setup, navigate to the `benchmarks` directory of
the model zoo and set environment variables pointing to the directory for the
dataset, pretrained model, and an output directory where log
files will be written.

```
# cd to the benchmarks directory in the model zoo
cd benchmarks

export OUTPUT_DIR=<directory where log files will be written>
export PRETRAINED_MODEL=<path to the pretrained model checkpoints>
export TF_WAVENET_DIR=<path to your clone of the TensorFlow WaveNet repo>
```

Start a model run by executing the launch script and passing args
specifying that we are running wavenet fp32 inference using TensorFlow,
along with a docker image that includes Intel Optimizations for TensorFlow
and the path to your clone of the WaveNet repo and the checkpoint
files that were downloaded.  We are also passing a couple of extra model args
for wavenet: the name of the checkpoint to use and the sample number.

```
python launch_benchmark.py \
    --precision fp32 \
    --model-name wavenet \
    --mode inference \
    --framework tensorflow \
    --socket-id 0 \
    --num-cores 1 \
    --docker-image <docker image> \
    --model-source-dir ${TF_WAVENET_DIR} \
    --checkpoint ${PRETRAINED_MODEL} \
    --output-dir ${OUTPUT_DIR} \
    -- checkpoint_name=model.ckpt-99 sample=8510
```

The logs are displayed in the console output as well as saved to a
file in the value of `${OUTPUT_DIR}`.

The tail of the log should look something like this:
```
Time per 500 Samples: ... sec
Samples / sec: ...
msec / sample: ...
Sample: 8000
Time per 500 Samples: ... sec
Samples / sec: ...
msec / sample: ...
Sample: 8500

Average Throughput of whole run: Samples / sec: ...
Average Latency of whole run: msec / sample: ...
Finished generating. The result can be viewed in TensorBoard.
Log file location: ${OUTPUT_DIR}/benchmark_wavenet_inference_fp32_20210601_143852.log
```
