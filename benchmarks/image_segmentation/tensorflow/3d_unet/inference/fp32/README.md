<!--- 0. Title -->
# 3D U-Net FP32 inference

<!-- 10. Description -->
## Description

This document has instructions for running 3D U-Net FP32 inference using
Intel-optimized TensorFlow.

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

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [fp32_inference.sh](fp32_inference.sh) | Runs inference with a batch size of 1 using the BraTS dataset and a pretrained model |

<!--- 50. AI Kit -->
## Run the model

Setup your environment using the instructions below, depending on if you are
using [AI Kit](/docs/general/tensorflow/AIKit.md):

<table>
  <tr>
    <th>Setup using AI Kit</th>
    <th>Setup without AI Kit</th>
  </tr>
  <tr>
    <td>
      <p>AI Kit does not currently support TF 1.15.2 models</p>
    </td>
    <td>
      <p>To run without AI Kit you will need:</p>
      <ul>
        <li>Python 3.6 or 3.7
        <li><a href="https://pypi.org/project/intel-tensorflow/1.15.2/">intel-tensorflow==1.15.2</a>
        <li>numactl
        <li>Keras==2.6.0rc3
        <li>numpy==1.16.3
        <li>nilearn==0.6.2
        <li>tables==3.4.4
        <li>nibabel==2.3.3
        <li>SimpleITK===1.2.0
        <li>h5py==2.10.0
        <li>A clone of the Model Zoo repo<br />
        <pre>git clone https://github.com/IntelAI/models.git</pre>
      </ul>
    </td>
  </tr>
</table>

Download the pre-trained model from the
[3DUnetCNN](https://github.com/ellisdg/3DUnetCNN/blob/ff5953b3a407ded73a00647f5c2029e9100e23b1/README.md#pre-trained-models)
repository. In this example, we are using the "Original U-Net" model, trained using the BRATS 2017 data.
Set the the `PRETRAINED_MODEL` env var as the path to the tumor_segmentation_model.h5 file.
```
wget https://www.dropbox.com/s/m99rqxunx0kmzn7/tumor_segmentation_model.h5
export PRETRAINED_MODEL=$(pwd)/tumor_segmentation_model.h5
```

After your environment is setup, set environment variables to the `DATASET_DIR`
and an `OUTPUT_DIR` where log files will be written. Ensure that you already have
the `PRETRAINED_MODEL` path set from the previous command.
Once the environment variables are all set, you can run the
[quickstart script](#quick-start-scripts).
```
# cd to your model zoo directory
cd models

export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>
export PRETRAINED_MODEL=<path to the pretrained model>

./quickstart/image_segmentation/tensorflow/3d_unet/inference/cpu/fp32/fp32_inference.sh
```

<!--- 90. Resource Links-->
## Additional Resources

* To run more advanced use cases, see the instructions [here](Advanced.md)
  for calling the `launch_benchmark.py` script directly.
* To run the model using docker, please see the [oneContainer](http://software.intel.com/containers)
  workload container:<br />
  [https://software.intel.com/content/www/us/en/develop/articles/containers/3d-unet-fp32-inference-tensorflow-container.html](https://software.intel.com/content/www/us/en/develop/articles/containers/3d-unet-fp32-inference-tensorflow-container.html).

