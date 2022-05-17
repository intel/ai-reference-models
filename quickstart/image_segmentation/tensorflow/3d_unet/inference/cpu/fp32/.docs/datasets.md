<!--- 30. Datasets -->
## Datasets

The following instructions are based on [BraTS2018 dataset preprocessing steps](https://github.com/ellisdg/3DUnetCNN/tree/update_to_brats18#tutorial-using-brats-data-and-python-3) in the [3D U-Net repository](https://github.com/ellisdg/3DUnetCNN/tree/update_to_brats18).
1. Download [BraTS2018 dataset](https://www.med.upenn.edu/sbia/brats2018/registration.html).
Please follow the steps to register and request the training and the validation data of the BraTS 2018 challenge.

2. Create a virtual environment and install the dependencies:
    ```
    # create a python3.6 based venv
    virtualenv --python=python3.6 brats18_env
    . brats18_env/bin/activate
    
    # install dependencies
    pip install intel-tensorflow==1.15.2
    pip install SimpleITK===1.2.0
    pip install keras==2.2.4
    pip install nilearn==0.6.2
    pip install tables==3.4.4
    pip install nibabel==2.3.3
    pip install nipype==1.7.0
    pip install numpy==1.16.3
    ```
    Install [ANTs N4BiasFieldCorrection](https://github.com/ANTsX/ANTs/releases/tag/v2.1.0) and add the location of the ANTs binaries to the PATH environmental variable:
    ```
    wget https://github.com/ANTsX/ANTs/releases/download/v2.1.0/Linux_Debian_jessie_x64.tar.bz2
    tar xvjf Linux_Debian_jessie_x64.tar.bz2
    cd debian_jessie
    export PATH=${PATH}:$(pwd)
    ```

3. Clone the [3D U-Net repository](https://github.com/ellisdg/3DUnetCNN/tree/update_to_brats18), and run the script for the dataset preprocessing:
    ```
    git clone https://github.com/ellisdg/3DUnetCNN.git
    cd 3DUnetCNN
    git checkout update_to_brats18
    
    # add the repository directory to the PYTHONPATH system variable
    export PYTHONPATH=${PWD}:$PYTHONPATH
    ```
    After downloading the dataset file `MICCAI_BraTS_2018_Data_Training.zip` (from step 1), place the unzipped folders in the `brats/data/original` directory.
    ```
    # extract the dataset
    mkdir -p brats/data/original && cd brats
    unzip MICCAI_BraTS_2018_Data_Training.zip -d data/original
    
    # import the conversion function and run the preprocessing:
    python
    >>> from preprocess import convert_brats_data
    >>> convert_brats_data("data/original", "data/preprocessed")
    
    # run training using the original UNet model to get `validation_ids.pkl` created in `brats` directory.
    python train.py 
    ```
After it finishes, set an environment variable to the path that contains the preprocessed dataset file `validation_ids.pkl`. 
```
export DATASET_DIR=/home/<user>/3DUnetCNN/brats
```
