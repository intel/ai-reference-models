# Introduction
## Jupyter Notebooks 
 
The Jupyter notebook helps users to analyze the performance benefit from using Intel® Optimizations for PyTorch with the IPEX package. The notebooks will show the comparison between the performance of stock PyTorch and Intel's optimizations for PyTorch.
  

| Analysis Type | Notebook | Notes|
| ------ | ------ | ------ |
|stock vs. Intel PyTorch | [benchmark_perf_comparison](benchmark_perf_comparison.ipynb)  | Compare performance between Stock and Intel PyTorch among different image-recognition models  |

# Miscellaneous Files
* profiling: 
  * profile_utils.py: python classes that help plotting and data processing
  * topo.ini : configurations for supported models
    * Configurations include the model name and precision
    * Users do not need to change those configurations. The just have to select from one of them using topo_index in the notebook.

# Getting Started
To run this experiment please install the latest version of miniforge with python 3.8. The installer can be [downloaded](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh)
## Environment Setup

##### Stock PyTorch 

1. Create conda env: `$conda create -n stock-torch python=3.7 matplotlib psutil pandas gitpython`
2. Activate the created conda env: `$conda activate stock-torch`
3. Install ipython: `(stock-torch) $conda install 'ipython<7.0.0' ipykernel`
3. Install stock PyTorch with a specific version: `(stock-torch) $conda install pytorch==1.7.0 torchvision cpuonly -c pytorch`
4. Install extra needed package: `(stock-torch) $pip install cxxfilt`
5. Deactivate conda env: `(stock-torch) $conda deactivate`
6. Register the kernel to Jupyter NB: `$~/miniforge/envs/stock-torch/bin/python  -m ipykernel install --user --name=stock-torch`

> NOTE: Please change the python path if you have a different folder path for anaconda3/miniforge. 
  After profiling, users can remove the kernel from Jupyter NB with `$jupyter kernelspec uninstall stock-torch`.

##### Intel PyTorch 

1. Create conda env: `$conda create -n intel-torch python=3.7 matplotlib psutil pandas gitpython`
2. Activate the created conda env: `$conda activate intel-torch`
3. Install ipython: `(intel-torch) $conda install 'ipython<7.0.0' ipykernel`
4. Update the mkl from Intel's channel" `(intel-torch) $conda update -c https://software.repos.intel.com/python/conda/ mkl`
5. Install Intel-optimized torch with a specific version: `(intel-torch) $conda install -c https://software.repos.intel.com/python/conda/ pytorch`
6. Install IPEX from intel's conda channel: `(intel-torch) $conda install -c https://software.repos.intel.com/python/conda/ intel-extension-for-pytorch`
7. Install Torchvision from PyTorch's channel: `(intel-torch) $conda install torchvision cpuonly -c pytorch`
8. Install extra needed package: `(intel-torch) $pip install cxxfilt`
9. Deactivate conda env: `(intel-torch) $conda deactivate`
10. Register the kernel to Jupyter NB: `$~/miniforge/envs/intel-torch/bin/python  -m ipykernel install --user --name=intel-torch`

> NOTE: Please change the python path if you have a different folder path for anaconda3/miniforge. 
  After profiling, users can remove the kernel from Jupyter NB with `$jupyter kernelspec uninstall intel-torch`.

## How to Run

1. Clone the Intel Model Zoo: `$git clone https://github.com/IntelAI/models.git`
2. Browse to the `models/docs/notebooks/perf_analysis/pytorch` folder
3. Launch Jupyter notebook: `$jupyter notebook --ip=0.0.0.0`
4. Follow the instructions to open the URL with the token in your browser
5. Click the notebook file `benchmark_perf_comparison.ipynb`.

> Note: For "stock v.s. Intel PyTorch" analysis type, please change your Jupyter notebook kernel to either "stock-torch" or "intel-torch"

6. Run through every cell of the notebook one by one

> NOTE: For "stock vs. Intel PyTorch" analysis type, in order to compare between stock and Intel-optimized PyTorch results in Section 5 of the notebook, users need to run all cells before the comparison section with both stock-torch and intel-torch kernels. 
