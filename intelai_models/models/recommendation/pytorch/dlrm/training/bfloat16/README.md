# DLRM MLPerf BF16 Training v0.7 Intel Submission
For License, Contribution and Code of conduct, please see here: https://github.com/facebookresearch/dlrm/tree/mlperf

## HW and SW requirements
### 1. HW requirements
| HW | configuration |
| -: | :- |
| CPU | CPX-6 @ 4 sockets/Node |
| DDR | 192G/socket @ 3200 MT/s |
| SSD | 1 SSD/Node @ >= 1T |

### 2. SW requirements
| SW |configuration  |
|--|--|
| GCC | GCC 8.3  |

## Steps to run DLRM

### 1. Install anaconda 3.0
```
   wget https://repo.continuum.io/archive/Anaconda3-5.0.0-Linux-x86_64.sh -O anaconda3.sh
   chmod +x anaconda3.sh
   ./anaconda3.sh -b -p ~/anaconda3
   ~/anaconda3/bin/conda create -n dlrm python=3.7

   export PATH=~/anaconda3/bin:$PATH
   source ./anaconda3/bin/activate dlrm
```
### 2. Install dependency packages
```
   pip install sklearn onnx tqdm lark-parser
   pip install -e git+https://github.com/mlperf/logging@0.7.0-rc2#egg=logging

   conda config --append channels intel
   conda install ninja pyyaml setuptools cmake cffi typing
   conda install intel-openmp mkl mkl-include numpy -c intel --no-update-deps
   conda install -c conda-forge gperftools
```
### 3. Clone source code and Install
(1) Install PyTorch and Intel Extension for PyTorch
```
   # clone PyTorch
   git clone https://github.com/pytorch/pytorch.git
   cd pytorch && git checkout tags/v1.5.0-rc3 -b v1.5-rc3
   git submodule sync && git submodule update --init --recursive

   # clone Intel Extension for PyTorch
   git clone https://github.com/intel/intel-extension-for-pytorch.git
   cd intel-extension-for-pytorch && git checkout tags/v0.2 -b v0.2
   git submodule update --init --recursive

   # install PyTorch
   cd {path/to/pytorch}
   cp {path/to/intel-pytorch-extension}/torch_patches/0001-enable-Intel-Extension-for-CPU-enable-CCL-backend.patch .
   patch -p1 < 0001-enable-Intel-Extension-for-CPU-enable-CCL-backend.patch
   python setup.py install

   # install Intel Extension for PyTorch
   cd {path/to/intel-pytorch-extension}
   python setup.py install
```
(2) Install oneCCL
```
   git clone https://github.com/oneapi-src/oneCCL.git
   cd oneCCL && git checkout 2021.1-beta07-1
   mkdir build && cd build
   cmake .. -DCMAKE_INSTALL_PREFIX=~/.local
   make install -j
```
(3) Install Torch CCL
```
   git clone https://github.com/intel/torch-ccl.git
   cd torch-ccl && git checkout 2021.1-beta07-1
   source ~/.local/env/setvars.sh
   python setup.py install
```
### 4. Prepare dataset
(1) Go to the Criteo Terabyte Dataset website(https://labs.criteo.com/2013/12/download-terabyte-click-logs/) and accept the terms of use. 
(2) Copy the data download URL in the following page, and run :
```
    mkdir <dir/to/save/dlrm_data> && cd <dir/to/save/dlrm_data>
    curl -O <download url>/day_{$(seq -s , 0 23)}.gz
    gunzip day_*.gz
```
(2) Please remember to replace  "<dir/to/save/dlrm_data>" to any path you want to download and save the dataset.
These raw data will be automatically pre-processed and saved as "day_*.npz" to <dir/to/save/dlrm_data> when you do the following steps at the first time.
After first running, the scripts below will automatically using pre-processed data.

### 5. Get DLRM model and run command for BF16 training

```
    git clone https://github.com/IntelAI/models.git
    cd models/recommendation/pytorch/dlrm/training/bf16
```
Run 32K global BS with 4 ranks on 1 node (1 CPX6-4s Node).
```
   # export DATA_PATH per your local environment
   export DATA_PATH=<dir/to/save/dlrm_data>

   # Clean resources (if have root or sudo authority)
   ./bench/cleanup.sh
 
   bench/dlrm_mlperf_4s_1n_cpx.sh
```
Run 32K global BS with 16 ranks on 4 nodes (4 CPX6-4s Nodes).
```
   # export DATA_PATH per your local environment
   export DATA_PATH=<dir/to/save/dlrm_data>

   # create `hostfile` per your local machines
   # Clean resources (if have root or sudo authority)
   mpiexec.hydra -np 4 -ppn 1 -f hostfile ./bench/cleanup.sh

   bench/dlrm_mlperf_16s_4n_cpx.sh
```
