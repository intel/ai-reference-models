# Install Intel® Extension for PyTorch
Prepare the environment, you may create a Python virtual enviromment `virtualenv` or `conda` prior to installing dependencies.

    # Install Intel® Extension for PyTorch
    pip install intel-extension-for-pytorch
    # Install torch,torchvision
    python -m pip install torch torchvision

## The following components are required by some PyTorch workloads. Only build them if indicated in the documentation for that workload.

    # Requirements:
    gcc >= 5
    Cmake >= 3.19.6

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


### Build torch-ccl 
    cd ..
    git clone https://github.com/intel-innersource/frameworks.ai.pytorch.torch-ccl.git
    # torch-ccl branch refer to https://github.com/intel-innersource/frameworks.ai.pytorch.ipex-cpu/blob/cpu-device/dependency_version.yml
    cd frameworks.ai.pytorch.torch-ccl && git checkout $target_branch
    git submodule sync 
    git submodule update --init --recursive
    python setup.py install 
