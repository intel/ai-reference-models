<!--- 50. Bare Metal -->
## Bare Metal

To run on bare metal, the following prerequisites must be installed in your environment:
* Python 3.6 or 3.7
* [intel-tensorflow==1.15.2](https://pypi.org/project/intel-tensorflow/1.15.2/)
* numactl
* Keras==2.6.0rc3
* numpy==1.16.3
* nilearn==0.6.2
* tables==3.4.4
* nibabel==2.3.3
* SimpleITK===1.2.0
* h5py==2.10.0

Follow the [instructions above for downloading the BraTS dataset](#dataset).

1. Download the pretrained model from the
   [3DUnetCNN repo](https://github.com/ellisdg/3DUnetCNN/blob/ff5953b3a407ded73a00647f5c2029e9100e23b1/README.md#pre-trained-models).
   In this example, we are using the "Original U-Net" model, trained using the
   BraTS 2017 data.

2. Download and untar the model package. Set environment variables for the path
   to your `DATASET_DIR`, `PRETRAINED_MODEL` and `OUTPUT_DIR` (where log files
   will be written), and then run the [quickstart script](#quick-start-scripts).

   ```
   wget <package url>
   tar -xzf <package name>
   cd <package dir>

   export DATASET_DIR=<path to the BraTS dataset>
   export PRETRAINED_MODEL=<Path to the downloaded tumor_segmentation_model.h5 file>
   export OUTPUT_DIR=<directory where log files will be written>
   # For a custom batch size, set env var `BATCH_SIZE` or it will run with a default value.
   export BATCH_SIZE=<customized batch size value>

   ./quickstart/fp32_inference.sh
   ```
