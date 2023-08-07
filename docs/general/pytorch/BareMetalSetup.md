# Install Intel® Extension for PyTorch
pip install intel-extension-for-pytorch==2.0.0

## The following components are required by some PyTorch workloads. Only build them if indicated in the documentation for that workload. 

### Prepare the environment:
    gcc >= 5
    Cmake >= 3.19.6
    wget https://repo.continuum.io/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh -O miniconda.sh
    chmod +x miniconda.sh
    ./miniconda.sh -b -p ~/miniconda
    ./miniconda/bin/conda create -yn pytorch
    export PATH=~/miniconda/bin:$PATH
    source ./miniconda/bin/activate pytorch
    pip install sklearn onnx
    pip install lark-parser hypothesis
    conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses psutil
    export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
    export work_space=/home/sdp  (you can get the summary.log in this path where the models performance and accuracy write) 
    
    # Install torch,torchvision
    python -m pip install torch==2.0.0 torchvision==0.15.1
    
### Install jemalloc
    Install jemalloc either using conda or from source

    Using conda:
    conda install jemalloc

    From source:
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
    cd frameworks.ai.pytorch.torch-ccl && git checkout public_master
    git submodule sync 
    git submodule update --init --recursive
    python setup.py install 

