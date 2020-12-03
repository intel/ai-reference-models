# Docker

This directory contains [partials](partials) for constructing Dockerfiles for
various use cases in the model zoo. It uses a modified assembler (copied from the
[TensorFlow repo](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/dockerfiles)).
front-ended with a cli `model-builder` that runs the assembler in a `imz-tf-tools` image. The model-builder cli
will build the imz-tf-tools image using and the tools.Dockerfile (copied from [TensorFlow repo](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/dockerfiles/tools.Dockerfile)) if `imz-tf-tools` image is not found.

## specs/ directory

The [specs](specs) directory contains model manifests that are parsed by the model-builder 
when creating model packages, constructing model dockerfiles and building model images.
This directory is organized by framework. The `model-builder` commands like `init-spec`,
`models`, `make`, `build`, `generate-documentation`, etc have a `--framework` (or `-f`)
argument that specifies which framework folder the specs will be used from.

```
tools/docker/specs/
├── ml
│   ├── scikit-learn-census_spec.yml
│   └── ...
├── pytorch
│   ├── pytorch-resnet50-bf16-inference_spec.yml
│   └── ...
└── tensorflow
    ├── bert-large-bfloat16-inference_spec.yml
    └── ...
```

## partials/ directory 

Dockerfiles that have already been constructed from the partials are placed in the 
model repos under the [dockerfiles](dockerfiles) directory. Under this directory there
are subdirectories for model containers and data containers. These files should not be edited directly. 
Instead, regen these dockerfiles by making changes to the [partials](partials) and [specs](specs)
and then run the command `model-builder` to reconstruct the dockerfiles.

### Adding a new model

To add a new container, you'll need:
* A list of dependencies that need to be installed in the container (like pycoco, mpi, horovod, etc)
* Scripts for any other setup that needs to be done (such as running protoc)
* A model package (for model containers)

The model zoo's container partials are setup where
[intel-optimized-tensorflow is used as the base](partials/tensorflow/tensorflow-base.partial.Dockerfile),
dependencies common to each deep learning category are added on top of that, and
then finally the model package is added on top of that.

For example, to build the ResNet50v1.5 FP32 training container, the 
[resnet50v1-5-fp32-training_spec](specs/tensorflow/resnet50v1-5-fp32-training_spec.yml)
shows that it is constructed by using the Intel TensorFlow base, image recognition,
and the ResNet50 v1.5 FP32 training package.

The `model-builder` has a `init-spec` subcommand for initializing a new spec
yaml file when adding a new model. The
[documentation on generating a new spec file](/tools/README.md#generating-a-new-spec-file)
provides more information on how it works, but the general idea is that
the spec will get auto-generated as much as possible.

The auto-generated spec file has a `releases` section with `tag_specs` that specify which
slice sets to put together to create the dockerfile and container for the model.

The default slice sets that are included in the yaml file when using `init-spec` are: 
* intel-tf (the base partial that starts `FROM intel/intel-optimized-tensorflow`)
* the slice set for the deep learning category for the model (like image-recognition, object-detection, etc)
* the slice set for the model/precision/mode (which is is the slice set specified in the model spec).

If your model can be run with multiple instances or multiple nodes and it requires that
MPI and horovod are installed, then add the `{mpi-horovod}` slice set to the `tag_specs`
for both the dockerfile and versioned container in your model spec. For example, bert large
uses the `{mpi-horovod}` slice set:

<pre>
releases:
    versioned:
        tag_specs:
            - "{_TAG_PREFIX}{intel-tf}{language-modeling}<mark><b>{mpi-horovod}</b></mark>{bert-large-fp32-training}"
</pre>

The `versioned` release group means that it will be built with the
[default TensorFlow tag](tools/scripts/model-builder#L745)
that is used by the `model-builder`. If the model need to be built with
a specific version of TensorFlow, define a different group for it. For
example, there are some models that need TensorFlow 1.15.2, and those
use the group called `tf_1.15.2_containers`.

Adding a container for a new model may require adding partials for a new
deep learning category, if it's the first of its kind being added. The category
partials would contain installs and setup that is common to models in that category.
For example, the object detection category partials install `pycoco` tools and run
protoc on the object detection scripts from the TensorFlow models repo. Those
partials are also used in the [base_spec.yml](specs/tensorflow/base_spec.yml) to defined a "category
container".

Once there are partials added for the category you are using, the model
package can be added on top of that to create the model container. This is done
using the [model package partial](partials/model_package.partial.Dockerfile).
This partial simply adds the model package .tar.gz file to the predefined `${MODEL_WORKSPACE}` environment variable in the
container (which also extracts the tar) and sets the working directory to that
package location.

Finally, either update the [base_spec.yml](specs/tensorflow/base_spec.yml) or add a new model-specific
spec file to add slice sets for the partials that you have added. The spec file also defines 
build args for things like the name of the model package. Once slice sets have been added, 
update `releases:` section at the top of the file to specify which slice sets to put together 
when building packages, dockerfiles and images.
