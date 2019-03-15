# Downloading and Preprocessing the BraTS Dataset

## Background

All of the Intel Model Zoo's 3D U-Net benchmarking scripts and tutorials use the **preprocessed BraTS dataset** as documented [here](https://github.com/ellisdg/3DUnetCNN/blob/master/README.md).
In case you have trouble following the steps and running the scripts in that repository, here is a walkthrough of the process on Ubuntu 16.04.
It is our hope that this can save some time and frustration, but if there is any doubt, please refer to the official [repository](https://github.com/ellisdg/3DUnetCNN) and contact its owner or submit a [GitHub issue](https://github.com/ellisdg/3DUnetCNN/issues).

## Steps

1. **Request Access to the CBICA Image Processing Portal and Submit a "Data Request" Job**: 
   Follow the steps at [https://www.med.upenn.edu/sbia/brats2018/registration.html](https://www.med.upenn.edu/sbia/brats2018/registration.html) to create an account and submit a job for "BraTS'18: Data Request". 
   Your team name must be original and unique. Once you have successfully run the job and downloaded the results, you will obtain a zip file that, when extracted, contains a file called "REGISTRATION_STATUS.txt". 
   This file contains URLs for the 2018 BraTS'18 dataset.  

2. **Download the Raw Dataset (20GB required)**: Using the URL for the "complete training data", download the data:
   ```
   $ cd ~
   $ wget <complete_training_data_URL>/MICCAI_BraTS_2018_Data_Training.zip
   ```
   Note: Do not extract any of the .gz files contained in the resulting directory and subdirectories.

3. **Install Miniconda for Python3 (3GB required)**: 
   Python3 is required to run the preprocessing scripts. The Miniconda environment manager worked best in our tests, even though packages were installed with a mixture of conda and pip.
     - Follow the installation instructions [here](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html) and either restart your shell or source your .bashrc/.bash_profile so that the path to Miniconda is in your PATH variable
     - Test the installation: `$ conda list`
     - Create a new environment: `$ conda create --name 3dunet`
     - Activate the environment: `$ conda activate 3dunet`
     
4. **Install Python3 Packages Required for Preprocessing**:
   The required Python packages are 
   [Keras](https://keras.io/), 
   [scikit-learn](https://scikit-learn.org),
   [NiBabel](https://nipy.org/nibabel/), 
   [PyTables](https://www.pytables.org/), 
   [Nilearn](https://nilearn.github.io/), 
   [SimpleITK](http://www.simpleitk.org/), and 
   [Nipype](https://nipype.readthedocs.io). Many of these are neuroimaging tools.
   ```
    $ conda install keras scikit-learn
    $ pip install nibabel tables nilearn SimpleITK nipype
   ```
   
   To check that the TensorFlow installation has MKL-DNN enabled:
   ```
   $ python -c "import tensorflow; print(tensorflow.pywrap_tensorflow.IsMklEnabled())"
   ```

   This should return `True`, confirming that TensorFlow was built with MKL-DNN.

5. **Get [ANTs Version 2.1.0](https://github.com/ANTsX/ANTs/releases/tag/v2.1.0) (1.5GB required)**: For Linux, download the Debian_jessie_x64 archive, extract it, and add its binaries to the PATH environment variable.
   ```
   $ wget https://github.com/ANTsX/ANTs/releases/download/v2.1.0/Linux_Debian_jessie_x64.tar.bz2
   $ tar xvjf Linux_Debian_jessie_x64.tar.bz2
   $ export PATH=${PATH}:$(pwd)/debian_jessie
   ``` 

6. **Clone the 3DUnetCNN repository**: Also, add it to the PYTHONPATH environment variable.
   ```
   $ git clone https://github.com/ellisdg/3DUnetCNN.git
   $ cd 3DUnetCNN
   $ export PYTHONPATH=${PWD}:$PYTHONPATH
   ``` 

7. **Extract the raw BraTS data into `~/3DUnetCNN/brats/data/original`**: There should be two folders in `original`, one named `LGG` and one named `HGG`. 
   ```
   $ mkdir brats/data
   $ mkdir brats/data/original
   $ unzip ~/MICCAI_BraTS_2018_Data_Training.zip -d brats/data/original
   ```
8. **Run the Preprocessing Script**:
   ```
   $ cd brats
   $ python
   >>> from preprocess import convert_brats_data
   >>> convert_brats_data("data/original", "data/preprocessed")
   ```
   
9. **Run the Training Script**: The training script converts the preprocessed data to H5 format and creates two additional files required by the benchmarking script.
   Note: You do not _actually_ need to train the model, you just need the data files, so if your machine runs out of memory or training does not complete for some reason, it is probably alright! 
   ```
   $ python train.py
   ```
   When the script finishes, make sure you see the following files in `~/3DUnetCNN/brats/`:
     - `brats_data.h5` (~6GB)
     - `training_ids.pkl` (~1KB)
     - `validation_ids.pkl` (~1KB)
     
10. **Deactivate the Conda Environment**:
    ```
    $ conda deactivate
    ```

## Conclusion
Now you are ready to run the benchmarking script with the `--data-location` flag set to `/home/<user>/3DUnetCNN/brats`.
