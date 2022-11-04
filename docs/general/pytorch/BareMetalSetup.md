# Install Intel® Extension for PyTorch
pip install intel-extension-for-pytorch==1.12.300

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
    pip install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses psutil
    
    # Install torch,torchvision and torchaudio
    python -m pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1
    
### Install jemalloc
    conda install jemalloc

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
    cd frameworks.ai.pytorch.torch-ccl && git checkout ccl_torch_1.13
    git submodule sync 
    git submodule update --init --recursive
    python setup.py install 

