<!--- 50. baremetal_spr -->
## Download the pretrained model
Download the model pretrained frozen graph from the given link based on the precision of your interest. Please set `PRETRAINED_MODEL` to point to the location of the pretrained model file on your local system.
```
# FP32, BFloat16 & BFloat32 Pretrained model for inference:
#TODO: To be publish externally
/tf_dataset/pre-trained-models/dien/fp32/dien_fp32_static_mklgrus.pb

# FP32, BFloat16 & BFloat32 Pretrained model for accuracy:
#TODO: To be publish externally
/tf_dataset/pre-trained-models/dien/bfloat16/dien_fp32_dynamic_mklgrus.pb
```

## Run the model

After you've followed the instructions to [download the pretrained model](#download-the-pretrained-model)
and [prepare the dataset](#datasets) using real data, set environment variables to
specify the path to the pretrained model, the dataset directory, precision to run, and an output directory.

You can change the defaut values for the batch size by setting `BATCH_SIZE` environemnt variable. Otherwise the default values in the [quick start scripts](#quick-start-scripts) will be used.

```
# Set the required environment vars
export PRECISION=<specify the precision to run>
export PRETRAINED_MODEL=<path to the downloaded pretrained model file>
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
# Optional env vars
export BATCH_SIZE=<customized batch size value>
```

Navigate to the models directory to run any of the available benchmarks.
```
cd models
```
### Run real time inference (Latency):
```
./quickstart/image_recognition/tensorflow/resnet50v1_5/inference/cpu/inference_realtime_multi_instance.sh
```

### Run inference (Throughput):
```
./quickstart/image_recognition/tensorflow/resnet50v1_5/inference/cpu/inference_throughput_multi_instance.sh
```

### Run accuracy:
```
./quickstart/image_recognition/tensorflow/resnet50v1_5/inference/cpu/accuracy.sh
```
