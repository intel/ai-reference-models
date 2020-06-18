# Introduction
## Jupyter Notebooks 
 
These Jupyter notebooks help users to analyze the performance benefit from using Intel Optimizations for Tensorflow with the oneDNN library.

| Notebook | Notes|
| ------ | ------ |
| benchmark_perf_comparison.ipynb | Compare performance between Stock and Intel Tensorflow among different models  |
| benchmark_perf_timeline_analysis.ipynb | Analyze the performance benefit from oneDNN among different layers by using Tensorflow Timeline |  

## Miscellaneous Files

* images: image files that are used by notebooks and README  
* profiling: 
  * profile_utils.py: python classes that help plotting and data processing
  * patches: git patch files for enabling profiling for supported models
  * topo.ini : configurations for supported models  
    * Configurations include the pre-trained model download link and git patch for the model 
    * Users do not need to change those configurations

# Getting Started

## Environment Setup

### **Prerequisites**

> NOTE: No action required if users use Intel DevCloud as their environment. 
  Please refer to [Intel oneAPI DevCloud](https://intelsoftwaresites.secure.force.com/devcloud/oneapi) for Intel DevCloud.

 1. **Python3 Environment**
    
    Choose one of:
    -  **Anaconda**  
       Users can refer to the [installation link](https://docs.anaconda.com/anaconda/install/) for details.

    -  **Virtualenv**  
       Users can use below commands to install virtualenv on Ubuntu.  
       ```
       sudo apt update  
       sudo apt install python3-dev python3-pip  
       sudo pip3 install -U virtualenv  # system-wide install
       ```

    > NOTE: Select either Anaconda or Virtualenv and go through only one Environment Creation section below.

 2. **Jupyter Notebook**  
       Users can install via PIP by `$pip install notebook`.  
       Users can also refer to the [installation link](https://jupyter.org/install) for details.

### **Conda Environment Creation**  

#### 1. Intel oneAPI DevCloud

##### Stock TensorFlow

1. Create conda env: `$conda create -n stock-tensorflow python matplotlib ipykernel psutil pandas gitpython`
2. Activate the created conda env: `$source activate stock-tensorflow`
3. Install stock Tensorflow with a specific version: `(stock-tensorflow) $pip install tensorflow==2.1.0`
4. Install extra needed package: `(stock-tensorflow) $pip install cxxfilt`
5. Deactivate conda env: `(stock-tensorflow)$conda deactivate`
6. Register the kernel to Jupyter NB: `$~/.conda/envs/stock-tensorflow/bin/python -m ipykernel install --user --name=stock-tensorflow`

>  NOTE: Please change the python path if you have a different folder path for anaconda3.
   After profiling, users can remove the kernel from Jupyter NB with `$jupyter kernelspec uninstall stock-tensorflow`.

##### Intel TensorFlow

> NOTE: Intel-optimized Tensorflow is on DevCloud. However, users don't have access to install extra packages. 
  Therefore, we need to clone Intel Tensorflow into the user's home directory for installing extra packages.

1. Source oneAPI environment variables: `$source /opt/intel/inteloneapi/setvars.sh`
2. Create conda env: `$conda create --name intel-tensorflow --clone tensorflow`  
3. Activate the created conda env: `$source activate intel-tensorflow`  
4  Install extra needed package: `(intel-tensorflow) $pip install cxxfilt matplotlib ipykernel psutil pandas gitpython`  
5. Deactivate conda env: `(intel-tensorflow)$conda deactivate`  
6. Register the kernel to Jupyter NB: `$~/.conda/envs/intel-tensorflow/bin/python  -m ipykernel install --user --name=intel-tensorflow`  

> NOTE: Please change the python path if you have a different folder path for anaconda3. 
  After profiling, users can remove the kernel from Jupyter NB with `$jupyter kernelspec uninstall intel-tensorflow`.

#### 2. Linux with Intel oneAPI AI Analytics Toolkit  

> NOTE: Users can adopt most of the steps from Intel oneAPI DevCloud sections. We will list some different steps below.  

#### 3. Linux without Intel oneAPI AI Analytics Toolkit

##### Stock TensorFlow

1. Create conda env: `$conda create -n stock-tensorflow python matplotlib ipykernel psutil pandas gitpython`

> NOTE: If users want to use Tensorflow v1.14, they need to install python v3.5 by assigning `python=3.5`.

2. Activate the created conda env: `$conda activate stock-tensorflow`
3. Install stock tensorflow with a specific version: `(stock-tensorflow) $pip install tensorflow==2.1.0`

> NOTE: You can change the Tensorflow version to different one. We validated on v1.14 and v2.1.0.

4. Install extra needed package: `(stock-tensorflow) $pip install cxxfilt`
5. Deactivate conda env: `(stock-tensorflow)$conda deactivate`
6. Register the kernel to Jupyter NB: `$~/anaconda3/envs/stock-tensorflow/bin/python  -m ipykernel install --user --name=stock-tensorflow`

> NOTE: Please change the python path if you have a different folder path for anaconda3. 
  After profiling, users can remove the kernel from Jupyter NB with `$jupyter kernelspec uninstall stock-tensorflow`.

##### Intel TensorFlow

1. Create conda env: `$conda create -n intel-tensorflow python matplotlib ipykernel psutil pandas gitpython`  

> NOTE: If users want to use Tensorflow v1.14, they need to install python v3.5 by assigning `python=3.5`

2. Activate the created conda env: `$conda activate intel-tensorflow`
3. Install Intel-optimized tensorflow with a specific version: `(intel-tensorflow) $pip install intel-tensorflow==2.1.0`

> NOTE: You can change the Tensorflow version to a different one. We validated on v1.14 and v2.1.0.

4. Install extra needed package: `(intel-tensorflow) $pip install cxxfilt`
5. Deactivate conda env: `(intel-tensorflow)$conda deactivate`
6. Register the kernel to Jupyter NB: `$~/anaconda3/envs/intel-tensorflow/bin/python  -m ipykernel install --user --name=intel-tensorflow`

> NOTE: Please change the python path if you have a different folder path for anaconda3. 
  After profiling, users can remove the kernel from Jupyter NB with `$jupyter kernelspec uninstall intel-tensorflow`.

### **Virtualenv Environment Creation**  

> NOTE: Virtualenv is not applicable for Intel DevCloud or oneAPI AI Analytics Toolkit.

#### Stock TensorFlow  

1. Create virtual env: `$virtualenv  -p python3 ./venv-stock-tf`  
2. Activate the created virtualenv:  `$source ./venv-stock-tf/bin/activate`  
3. Install required packages:  `(venv-stock-tf)$pip install matplotlib ipykernel psutil pandas cxxfilt gitpython`  
4. Install intel tensorflow with specific version: `(venv-stock-tf)$pip install tensorflow==2.1.0`  
5. Deactivate virtualenv: `(venv-stock-tf)$deactivate`  
6. Register the kernel to Jupyter NB: `$venv-stock-tf/bin/python  -m ipykernel install --user --name=stock-tensorflow`  

> NOTE: After profiling, users can remove the kernel from Jupyter NB with `$jupyter kernelspec uninstall stock-tensorflow`.

#### Intel TensorFlow  

1. Create virtual env: `$virtualenv  -p python3 ./venv-intel-tf`  
2. Activate the created virtualenv:  `$source ./venv-intel-tf/bin/activate`  
3. Install required packages:  `(venv-intel-tf)$pip install matplotlib ipykernel psutil pandas cxxfilt gitpython`  
4. Install Intel-optimized tensorflow with a specific version: `(venv-intel-tf)$pip install intel-tensorflow==2.1.0`  
5. Deactivate virtualenv: `(venv-intel-tf)$deactivate`  
6. Register the kernel to Jupyter NB: `$venv-intel-tf/bin/python  -m ipykernel install --user --name=intel-tensorflow` 

> NOTE: After profiling, users can remove the kernel from Jupyter NB with `$jupyter kernelspec uninstall intel-tensorflow`.

## How to Run

1. Clone the Intel Model Zoo: `$git clone https://gitlab.devtools.intel.com/intelai/models.git`
2. Launch Jupyter notebook: `$jupyter notebook --ip=0.0.0.0`

> NOTE: Users don't need to apply step 2 on DevCloud Environment.   

3. Follow the instructions to open the URL with the token in your browser
4. Browse to the `models/docs/notebooks/perf_analysis` folder
5. Click the `benchmark_perf_comparison.ipynb` or `benchmark_perf_timeline_analysis.ipynb` file
6. Change your Jupyter notebook kernel to either "stock-tensorflow" or "intel-tensorflow" (highlighted in the diagram below)
    <br><img src="images/jupyter_kernels.png" width="300" height="300"><br>
7. Run through every cell of the notebook one by one

> NOTE: In order to compare between stock and Intel-optimized TF results in section "Analyze TF Timeline results among Stock and Intel Tensorflow", users need to run all cells before the comparison section with both stock-tensorflow and intel-tensorflow kernels. 


