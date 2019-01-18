# C-API inference application
The `infer_ocr_recognition` application serves for running C-API based
inference benchmark for CRNN-CTC model.

# How to build C-API application
In order to build C-API inference application follow these three steps:
1. build paddle.
2. build paddle's target `inference_lib_dist`.
3. build capi inference application.
4. prepare models for inference.
5. prepare dataset

Each one will be shortly described below.
## 1. Build paddle
Do it as you usually do it. In case you never did it, here are example instructions:
```
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle
mkdir build
cd build
cmake .. -DWITH_DOC=OFF -DWITH_GPU=OFF -DWITH_DISTRIBUTE=OFF -DWITH_MKLDNN=ON -DWITH_GOLANG=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DWITH_PROFILER=ON -DWITH_FLUID_ONLY=ON -DON_INFER=ON
make -j <num_cpu_cores>
```
## 2. Build paddle's target `inference_lib_dist`
While still staying in `/path/to/Paddle/build`, build the target `inference_lib_dist`:
```
make -j <num_cpu_cores> inference_lib_dist
```
Now a directory named `fluid_install_dir` should exist in the build directory.
Remember that path.

## 3. Build C-API inference application
Go to the location of this README.md and execute:
```
mkdir build
cd build
cmake .. -DPADDLE_ROOT=/path/to/Paddle/build/fluid_install_dir
make
```

## 4. Prepare models
Unpack the models you need:

`models/ocr_recognition/paddlepaddle/CRNN-CTC/CRNN-CTC_model.tar.gz`
is for inference with accuracy measuring.

`models/ocr_recognition/paddlepaddle/CRNN-CTC/CRNN-CTC_model_noacc.tar.gz`
is for inference with performance measuring.

## 5. Prepare dataset
Download the dataset
```
cd $PATH_FOR_DATASET
wget http://paddle-ocr-data.bj.bcebos.com/data.tar.gz
tar -zxvf data.tar.gz
```

# Run
If everything builds successfully, you can inference.

To run inference with a model for accuracy measuring, use the
`CRNN-CTC_model` model and option `--with_labels=1`.

An exemplary command:
```
./infer_ocr_recognition \
  --infer_model=<path_to_directory_with_model> \
  --data_list=<path_to_dataset>/test.list \
  --data_dir=<path_to_dataset>/test_images \
  --paddle_num_threads=<num_cpu_cores> \
  --use_mkldnn=true \
  --batch_size=1 \
  --iterations=100  \
  --skip_batch_num=5 \
  --enable_graphviz=1 \
  --with_labels=1 \
```
To run inference with a model for performance benchmarking (without accuracy),
use the `CRNN-CTC_model_noacc` model and the option `--with_labels=0`.

To add profiling, use the `--profile` option.

To generate .dot files with data flow graphs, use option `--enable_graphviz=1`.

To run inference without running passes, use option `--skip_passes`.

