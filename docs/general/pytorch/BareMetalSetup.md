# Install Intel® Extension for PyTorch
Prepare the environment, you may create a Python virtual enviromment `virtualenv` or `conda` prior to installing dependencies.

    # Install Intel® Extension for PyTorch:
    python -m pip install intel-extension-for-pytorch
    
    # Install torch,torchvision:
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

    # Install oneccl:
    python -m pip install oneccl_bind_pt --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/

## Install generic dependencies
Make sure the following components are installed in your environment :

    # Requirements:
    gcc >= 5
    Cmake >= 3.19.6
    build-essential 
    ca-certificates 
    git 
    wget 
    make 
    cmake 
    autoconf 
    bzip2 
    tar
    numactl 
    libegl1-mesa 

### Install jemalloc
    Install jemalloc either using conda or from source

    Using conda:
    conda install -y jemalloc=5.2.1 -c conda-forge

    From source:
    cd ..
    git clone  https://github.com/jemalloc/jemalloc.git    
    cd jemalloc
    git checkout 5.2.1
    ./autogen.sh
    ./configure --prefix=your_path(eg: /home/tdoux/tdoux/jemalloc/)
    make
    make install

### Build tcmalloc 
    Install tcmalloc using conda

    Using conda:
    conda install -y gperftools -c conda-forge 

### Build torch-ccl 
    cd ..
    git clone https://github.com/intel-innersource/frameworks.ai.pytorch.torch-ccl.git
    # torch-ccl branch refer to https://github.com/intel-innersource/frameworks.ai.pytorch.ipex-cpu/blob/cpu-device/dependency_version.yml
    cd frameworks.ai.pytorch.torch-ccl && git checkout $target_branch
    git submodule sync 
    git submodule update --init --recursive
    python setup.py install 
