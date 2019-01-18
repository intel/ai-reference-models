# C-API inference application
The `infer_image_classification` application serves for running C-API based
inference benchmark for image classification models: ResNet50, SE-ResNeXt50
and MobileNet-v1.

# How to build C-API application
In order to build C-API inference application follow these three steps:
1. build paddle.
2. build paddle's target `fluid_lib_dist`.
3. build capi inference application.
4. prepare models for inference.

## 1. Build paddle
Do it as you usually do it. In case you never did it, here are example instructions:
```
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle
mkdir build
cd build
cmake .. -DWITH_DOC=OFF -DWITH_GPU=OFF -DWITH_DISTRIBUTE=OFF -DWITH_MKLDNN=ON -DWITH_GOLANG=OFF -DWITH_SWIG_PY=ON -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DWITH_TIMER=OFF -DWITH_PROFILER=ON -DWITH_FLUID_ONLY=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DON_INFER=ON
make -j <num_cpu_cores>
```
## 2. Build paddle's target `inference_lib_dist`
While still staying in `/path/to/Paddle/build`, build the target `inference_lib_dist`:
```
make -j <num_cpu_cores> inference_lib_dist
```
Now a directory should exist in build directory named `fluid_install_dir`. Remember that path.
## 3. Build C-API inference application
Now go to where this README.md is and execute
```
mkdir build
cd build
cmake .. -DPADDLE_ROOT=/path/to/Paddle/build/fluid_install_dir
make
```
## 4. Prepare models
Unpack the models you need:
`models/image_recognition/paddlepaddle/ResNet50/ResNet50_baidu.tar.gz`
`models/image_recognition/paddlepaddle/MobileNet-v1/MobileNet-v1_baidu.tar.gz`
`models/image_recognition/paddlepaddle/SE-ResNeXt50/SE-ResNeXt50_baidu.tar.gz`

# Run
If everything builds successfully, you can use a script like below to run inference:
```
#!/bin/bash
./infer_image_classification \
        --infer_model=<path_to_directory_with_model> \
        --data_list=<path_to>/ILSVRC2012/val_list.txt \
        --data_dir=<path_to>/ILSVRC2012/ \
        --paddle_num_threads=<num_cpu_cores> \
        --batch_size=50 \
        --skip_batch_num=0 \
        --iterations=10  \
        --profile \
        --use_mkldnn
```
The command as above requires the inference model (passed via the `infer_model`
option) to return accuracy as the second output and model parameters to be
stored in separate files.

To run inference on a model without accuracy, with parameters stored
in a single file, and with input image size 318x318, run:
```
#!/bin/bash
./infer_image_classification \
        --infer_model=<path_to_directory_with_model> \
        --data_list=<path_to>/ILSVRC2012/val_list.txt \
        --data_dir=<path_to>/ILSVRC2012/ \
        --paddle_num_threads=<num_cpu_cores> \
        --batch_size=50 \
        --skip_batch_num=0 \
        --iterations=10  \
        --profile \
        --use_mkldnn \
        --with_labels=0 \
        --one_file_params=1 \
        --resize_size=318 \
        --crop_size=318
```
