# Model Zoo packaging and containers

The [model-builder script](/tools/scripts/model-builder) is used to
build model packages and containers. It is intended for use by CI/CD systems,
but it can also be used by engineers to facilitate adding new packages/containers
and for local testing.

## Model Builder Setup

**Prerequisites:**
* Docker (this is already a requirement for running Intel Model Zoo scenarios in Docker)
* a clone of the [IntelAI/models repo](https://github.com/IntelAI/models) (`git clone https://github.com/IntelAI/models`)

**Setup steps**

1. Add the model-builder directory to your `PATH`:
   ```
   export PATH=${PWD}/tools/scripts:$PATH
   ```
   > If you do not add this directory to your `PATH`, you will need to
   > provide the full path when calling the `model-builder`.

2. Navigate to the `models` repo to execute the `model-builder`. Use the
   `--help` flag to see a list of all the options.
   ```
   cd <path-to-models-repo>
   model-builder --help
   ```

   Common `model-builder` commands are:
   * `model-builder models`: See a list the available models
   * `model-builder make <model>`: Builds the model package tar file,
     constructs the dockerfile, and builds the docker image for the
     specified model. The model specified should be one of the models
     listed from `model-builder models`.
   * `model-builder build -r <release group> <model>`: Builds the docker
     container for the specified model. If no `--release-group` or `-r`
     argument is provided, the model builder defaults to use the
     TensorFlow release groups. Release groups are defined in the model
     spec yml files.
   * `model-builder init-spec <new model>`: Creates a new spec file for
     the specified model. See the [section below](#steps-to-add-a-new-model-spec)
     for full instructions on using this command.


For details on the model-builder script see the
[model builder advanced documentation](ModelBuilderAdvanced.md).

## Steps to add a new model spec

Each model has a [spec file](docker/specs) which includes information
that is used to build model packages, construct dockerfiles, and build
images.

The `model-builder` can be used to initialize a new model spec file, which
will include as much information that can be auto-generated. Manual
updates may need to be made for partials required by the model,
additional build args, and the list of model package files may need to
be tweaked if the code organization in the model zoo repo is non-standard.
It's assumed that the model which you are creating a spec for has already
been added to the model zoo repo.

Follow the [model builder setup above](#model-builder-setup) prior to
using these steps.

1. Use the following command to initialize a new spec file for your model.
   If a pretrained model needs to be included in the model package
   (this will be the case for most inference models), then set the
   `MODEL_URL` path to the intel-optimized-tensorflow gcloud release
   bucket URL for the pretrained model. Setting `MODEL_URL` can be
   omitted if there is no pretrained model that needs to be included
   in the model package.
   ```
   MODEL_URL=<gcloud url> model-builder init-spec <spec name>
   ```

   Example for inception v4 FP32 inference:
   ```
    MODEL_URL=https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/inceptionv4_fp32_pretrained_model.pb model-builder init-spec inceptionv4-fp32-inference
   ```

   The spec name should be formatted like `modelname-precision-mode`
   (for example: `rfcn-fp32-inference` or `resnet50v1-5-fp32-training`)
   where the model name matches the name of a model in the
   [model zoo repo](https://gitlab.devtools.intel.com/intelai/models)
   except with underscores changed to dashes.

   When the `model-builder init-spec` command is run, it will create a yaml
   file in [tools/docker/specs folder](docker/specs)
   named `<spec name>_spec.yml`. If you have not already created
   [quickstart scripts](https://gitlab.devtools.intel.com/intelai/models/-/tree/develop/quickstart)
   for your model, the `init-spec` command also adds a quickstart template
   script for your model.

   The `model-builder init-spec` command will also copy markdown snippets from `tools/docker/docs`
   to `quickstart/<use_case>/tensorflow/<model>/<mode>/<precision>/.docs`. These snippets are generic and 
   should be updated to include model specific information.

   > **Optional:** If additional library installs (like pip installs) or
   > setup is needed to run your model, then manual updates will need to
   > be made to define other partials. Also, if your model does not
   > follow the standard model zoo directory structure, then you will
   > need to update the list of `files` in the spec yaml to list any
   > other files that are needed to run your model. See the
   > [docker documentation](/tools/docker/README.md) for details.

2. Test the new spec file by running the `model-builder make` command:
   ```
   model-builder make <spec name>
   ```
   After this command runs, you'll have:
   * A model README.md in your `quickstart/<use_case>/tensorflow/<model>/<mode>/<precision>/`
     that was built using markdown snippets from `quickstart/<use_case>/tensorflow/<model>/<mode>/<precision>/.docs`. 
   * A model package tar file your `<models repo>/output` directory
     (this can be used for testing the model on bare metal).
   * A model dockerfile in your `<models repo>/dockerfiles` directory.
     *Commit this file to your branch along with your model code.*
   * A local docker image (check `docker images` to see the list of
     images on your machine. This image can be used to locally test
     running the model in a container).

3. When the model spec yaml is complete, create an MR that adds
   the new file to the repo.
