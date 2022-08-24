# Install Pytorch* and Intel® Extension for PyTorch*
Follow [these instructions ](https://intel.github.io/intel-extension-for-pytorch/1.12.0/tutorials/installation.html) to install pytorch and IPEX via pip.

## The following components are required by some PyTorch* workloads. Only build them if indicated in the documentation for that workload. 

### Prepare the environment:
    gcc >= 5
    Cmake >= 3.19.6
    wget https://repo.continuum.io/archive/Anaconda3-5.0.0-Linux-x86_64.sh -O anaconda3.sh
    chmod +x anaconda3.sh
    ./anaconda3.sh -b -p ~/anaconda3
    ./anaconda3/bin/conda create -yn pytorch
    export PATH=~/anaconda3/bin:$PATH
    source ./anaconda3/bin/activate pytorch
    pip install sklearn onnx
    pip install lark-parser hypothesis
    conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses psutil
    export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
    export work_space=/home/sdp  (you can get the summary.log in this path where the models performance and accuracy write)   

### Build jemalloc
    cd ..
    git clone  https://github.com/jemalloc/jemalloc.git    
    cd jemalloc
    git checkout c8209150f9d219a137412b06431c9d52839c7272
    ./autogen.sh
    ./configure --prefix=your_path(eg: /home/tdoux/tdoux/jemalloc/)
    make
    make install

### Build tcmalloc 
    wget https://github.com/gperftools/gperftools/releases/download/gperftools-2.7.90/gperftools-2.7.90.tar.gz
    tar -xzf gperftools-2.7.90.tar.gz 
    cd gperftools-2.7.90
    ./configure --prefix=$HOME/.local
    make
    make install

### Build vision
    cd ..
    git clone https://github.com/pytorch/vision
    cd vision
    python setup.py install

### Build torch-ccl 
    cd ..
    git clone https://github.com/intel-innersource/frameworks.ai.pytorch.torch-ccl.git
    cd frameworks.ai.pytorch.torch-ccl && git checkout torch-ccl-1.12-rc1
    git submodule sync 
    git submodule update --init --recursive
    python setup.py install 

