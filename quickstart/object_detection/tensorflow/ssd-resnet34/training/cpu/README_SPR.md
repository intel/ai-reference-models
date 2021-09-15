<!--- 0. Title -->
# TensorFlow SSD-ResNet34 training

<!-- 10. Description -->

This document has instructions for running SSD-ResNet34 training
using Intel-optimized TensorFlow.


## Model Package

The model package includes the Dockerfile and scripts needed to build and
run SSD-ResNet34 training in a container.
```
tf-spr-ssd-resnet34-training
├── README.md
├── build.sh
├── licenses
│   ├── LICENSE
│   └── third_party
├── model_packages
│   └── tf-spr-ssd-resnet34-training.tar.gz
├──tf-spr-ssd-resnet34-training.Dockerfile
└── run.sh
```

<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| `training.sh` | Uses mpirun to execute 2 processes with 1 process per socket with a batch size of 32 for the specified precision (fp32 or bfloat16). Logs for each instance are saved to the output directory. |

<!--- 30. Datasets -->
## Datasets

SSD-ResNet34 training uses the COCO training dataset. Use the following instructions
to download and preprocess the dataset.

1.  Download and extract the 2017 training images and annotations for the
    [COCO dataset](http://cocodataset.org/#home):
    ```bash
    export MODEL_WORK_DIR=$(pwd)

    # Download and extract train images
    wget http://images.cocodataset.org/zips/train2017.zip
    unzip train2017.zip

    # Download and extract annotations
    wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip annotations_trainval2017.zip
    ```

2.  Since we are only using the train and validation dataset in this example,
    we will create an empty directory and empty annotations json file to pass
    as the test directories in the next step.
    ```bash
    # Create an empty dir to pass for validation and test images
    mkdir empty_dir

    # Add an empty .json file to bypass validation/test image preprocessing
    cd annotations
    echo "{ \"images\": {}, \"categories\": {}}" > empty.json
    cd ..
    ```

3. Use the [TensorFlow models repo scripts](https://github.com/tensorflow/models)
   to convert the raw images and annotations to the TF records format.
   ```
   git clone https://github.com/tensorflow/models.git tf_models
   cd tf_models
   git checkout 7a9934df2afdf95be9405b4e9f1f2480d748dc40
   cd ..
   ```

4. Install the prerequisites mentioned in the
   [TensorFlow models object detection installation doc](https://github.com/tensorflow/models/blob/v2.3.0/research/object_detection/g3doc/installation.md#dependencies)
   and run [protobuf compilation](https://github.com/tensorflow/models/blob/v2.3.0/research/object_detection/g3doc/installation.md#protobuf-compilation)
   on the code that was cloned in the previous step.

5. After your envionment is setup, run the conversion script:
   ```
   cd tf_models/research/object_detection/dataset_tools/

   # call script to do conversion
   python create_coco_tf_record.py --logtostderr \
         --train_image_dir="$MODEL_WORK_DIR/train2017" \
         --val_image_dir="$MODEL_WORK_DIR/empty_dir" \
         --test_image_dir="$MODEL_WORK_DIR/empty_dir" \
         --train_annotations_file="$MODEL_WORK_DIR/annotations/instances_train2017.json" \
         --val_annotations_file="$MODEL_WORK_DIR/annotations/empty.json" \
         --testdev_annotations_file="$MODEL_WORK_DIR/annotations/empty.json" \
         --output_dir="$MODEL_WORK_DIR/output"
    ```

    The `coco_train.record-*-of-*` files are what we will use in this training example.
    Set the output of the preprocessing script (`export DATASET_DIR=$MODEL_WORK_DIR/output`)
    when running quickstart scripts.

## Build the container

The SSD-ResNet34 training package has scripts and a Dockerfile that are
used to build a workload container that runs the model. This container
uses the TensorFlow SPR container as it's base, so ensure that you have built
the `tensorflow-spr.tar.gz` container prior to building this model container.

Use `docker images` to verify that you have the base container built. For example:
```
$ docker images | grep tensorflow-spr
model-zoo     tensorflow-spr    a5f08b0abf25     23 seconds ago   1.58GB
```

To build the SSD-ResNet34 training container, extract the package and
run the `build.sh` script.
```
# Extract the package
tar -xzf tf-spr-ssd-resnet34-training.tar.gz
cd tf-spr-ssd-resnet34-training

# Build the container
./build.sh
```

After the build completes, you should have a container called
`model-zoo:tf-spr-ssd-resnet34-training` that will be used to run the model.

## Run the model

After you've followed the instructions to [build the container](#build-the-container)
and [prepare the dataset](#datasets), use the `run.sh` script from the container
package to run SSD-ResNet34 training in docker. Set environment variables to
specify the dataset directory, precision to run, and
an output directory. By default, the `run.sh` script will run the
`training.sh` quickstart script.
```
# Navigate to the container package directory
cd tf-spr-ssd-resnet34-training

# Set the required environment vars
export PRECISION=<specify the precision to run>
export DATASET_DIR=<path to the dataset>
export OUTPUT_DIR=<directory where log files will be written>

# Run the container with training.sh quickstart script
./run.sh
```

<!--- 80. License -->
## License

Licenses can be found in the model package, in the `licenses` directory.

