# C-API inference application
The `infer_object_detection` application serves for running C-API based
inference benchmark for MobileNet-SSD model.

# How to build C-API application
In order to build C-API inference application follow these three steps:
1. build paddle.
2. build paddle's target `inference_lib_dist`.
3. build capi inference application.
4. prepare models for inference.

Each one will be shortly described below.
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
Now a directory named `fluid_install_dir` should exist in the build directory.
Remember that path.

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

`models/object_detection/paddlepaddle/MobileNet-SSD/MobileNet-SSD_pascalvoc.tar.gz`
is for inference with accuracy measuring.

`models/object_detection/paddlepaddle/MobileNet-SSD/MobileNet-SSD_pascalvoc_no_acc.tar.gz`
is for inference with performance measuring.

# Run
If everything builds successfully, you can run inference.

To run inference with a model for accuracy measuring, use the
`MobileNet-SSD_pascalvoc` model and option `--with_labels=1`.

An exemplary command:
```
./infer_object_detection \
        --infer_model=<path_to_directory_with_model> \
        --data_list=<path_to_pascalvoc_dataset>/test.txt \
        --label_list=<path_to_pascalvoc_dataset>/label_list \
        --data_dir=<path_to_pascalvoc_dataset> \
        --paddle_num_threads=<num_cpu_cores> \
        --batch_size=50 \
        --skip_batch_num=10 \
        --iterations=1000  \
        --use_mkldnn=1 \
	--with_labels=1 \
	--one_file_params=0
```
To run inference with a model for performance benchmarking (without accuracy),
use the `MobileNet-SSD_pascalvoc_no_acc` model and the option `--with_labels=0`.

To add profiling, use the `--profile` option.

To generate .dot files with data flow graphs, use option `--enable_graphviz=1`.

To resize images to a size different thant 300x300 before inference, use option
`--resize_size=512`.

To run inference without running passes, use option `--skip_passes`.

To run inference on a model with parameters stored in a single file, use option
`--one_file_params=1`.
