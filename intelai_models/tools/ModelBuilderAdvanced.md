# Model Builder - Advanced

This document provides details and information on more advanced usages
of the [model-builder script](/tools/scripts/model-builder). For
basic usage instructions, see the [README.md](README.md) file.

## Global Options

Running `model-builder options` or `model-builder --help` will show the global options

```
-h --help --verbose --dry-run
```

### --verbose

This global option will output the internal call to docker run as well as debug output from assembler.py (which is invoked within docker).

### --dry-run

This global option will output what model-builder will execute; it shows just the docker run calls that would occur.

## Custom Settings

There are environment variables that can be set prior to running the
model-builder script in order provide custom settings for the model
packages and containers.

| Variable | Default value | Description |
|----------|---------------|-------------|
| `MODEL_PACKAGE_DIR` | `../output` | Directory where model package .tar.gz files are located |
| `LOCAL_REPO` | `model-zoo` | Local images will be built as `${LOCAL_REPO}:tag`. Tags are defined by the spec yml files |
| `TENSORFLOW_TAG` | `latest` | Tag of the intel-optimized-tensorflow image to use as the base for versioned containers |
| `TAG_PREFIX` | `${TENSORFLOW_TAG}` | Prefix used for the image tags (typically this will be the TF version) |
| `MODEL_WORKSPACE` | `/workspace` | Location where the model package will be extracted in the model container |
| `IMAGE_LIST_FILE` | None | Specify a file path where the list of built image/tags will be written. This is used by automated build scripts. |
| `TEST_DIR` | None | Specify the root directory for testing docker partials. See run\_tests\_path argument to assembler for more details |


You can set these environment variables to customize the model-builder
settings. For example:

```
MODEL_PACKAGE_DIR=/tmp/model_packages model-builder
```
`model-builder` supports `CentOS` or `Ubuntu` based TensorFlow tags. Use the environment variable `TENSORFLOW_TAG` to specify the tag for the base `intel-optimized-tensorflow` image.
For example:
```
TENSORFLOW_TAG=centos-8 model-builder make mobilenet-v1-int8-inference
```
You can find the list of supported models on CentOS in `tools/docker/specs/centos`.
## Documentation text replacement

When `init-spec` is run, model spec yaml file's documentation section has
a `text_replace` dictionary that defines keyword and value pairs that
will be replaced when the final README.md is generated. The final README.md
can be generated using either `model-builder generate-documentation <model>`
or `model-builder make <model>`. The `text_replace` section is _optional_,
and if it doesn't exist then no text replacement will happen when
documentation is generated.

By default, when `init-spec` is run, the following text replacement
options will be defined in the model's spec yaml file:

| Keyword | Value |
|---------|-------|
| `<model name>` | The model's name formatted to be written in sentences (like `ResNet50` or `SSD-MobileNet`) |
| `<precision>` | The model's precision formatted to be written in sentences (like `FP32` or `Int8`) |
| `<mode>` | The mode for the model package/container (`inference` or `training`)
| `<use_case>` | The model's use case formatted as it is in the model zoo directory structure (like `image_recognition` or `object_detection`) |
| `<model-precision-mode>` | The model spec name, which consists of the model name, precision, and mode, as it's formatted in file names (like `resnet50-fp32-inference`) |

An example of what this looks like in the spec yaml is below:
```
    documentation:
        ...
        text_replace:
            <mode>: inference
            <model name>: SSD-ResNet34
            <precision>: FP32
            <use case>: object_detection
            <package url>:
            <package name>: ssd-resnet34-fp32-inference.tar.gz
            <package dir>: ssd-resnet34-fp32-inference
            <docker image>:
```

> Note: Please make sure to fill in the package url and docker image
> once the they have been uploaded and pushed to a repo.

After `init-spec` is run, these values can be changed (for example, if
the `<model name>` is not formatted correctly).

The [documentation fragments](/tools/docker/docs) use the keywords.
For example, [title.md](/tools/docker/docs/title.md) has:
```
<!--- 0. Title -->
# <model name> <precision> <mode>
```

When the documentation is generated, the text subsitution will happen and
the generated README.md will have the values filled in:
```
<!--- 0. Title -->
# SSD-ResNet34 FP32 inference
```

## Framework argument

The `--framework <framework>` (or `-f`) flag to the `model-builder` script
refers the names fo the folders in the [specs folder](/tools/docker/specs):
```
tools/docker/specs/
├── k8s
├── ml
├── pytorch
└── tensorflow
```

See a list of the available frameworks using:
```
$ model-builder frameworks
k8s ml pytorch tensorflow
```

The model-builder script uses this value for the `--spec_dir` and `--framework`
args when calling the [assembler.py](/tools/docker/assembler.py).

The `--framework` (or `-f`) flag applies to the following model-builder subcommands:
* build (e.g `model-builder build -f ml xgboost`)
* generate-dockerfile (e.g. `model-builder generate-dockerfile -f ml`)
* generate-documentation (e.g. `model-builder generate-documentation -f tensorflow`)
* generate-deployment (e.g. `model-builder generate-deployment -f k8s`)
* images (e.g `model-builder images -f pytorch` or `model-builder images -f ml`)
* init-spec (e.g. `model-builder init-spec -f tensorflow inceptionv4-fp32-inference`)
* make (e.g. `model-builder make -f pytorch pytorch-resnet50-bfloat16-inference`)
* models (e.g. `model-builder models -f pytorch`)
* package (e.g. `model-builder package -f pytorch pytorch-resnet50-bfloat16-inference`)
* packages (e.g. `model-builder packages -f tensorflow`)
* run-test-suite (e.g. `model-builder run-test-suite -c generate-dockerfile -f pytorch`)

> If no `--framework <framework>` (or `-f`) flag is passed to the
> `model-builder`, the default behavior will be to use `tensorflow`.

## Under the hood of the subcommands

### Building packages

The model-builder command will build packages by calling docker run on the imz-tf-tools container passing
in arguments to assembler.py. This internal call looks like the following:

```
docker run --rm -u 503:20 -v <path-to-models-repo>/tools/docker:/tf -v $PWD:/tf/models imz-tf-tools python3 assembler.py --release dockerfiles --build_packages --model_dir=models --output_dir=models/output
```

For single targets such as `bert-large-fp32-training` the model-builder adds an argument:

```
--only_tags_matching=.*bert-large-fp32-training$
```

### Constructing Dockerfiles

The model-builder command will construct Dockerfiles by calling docker run on the imz-tf-tools container passing
in arguments to assembler.py. This internal call looks like the following:

```
docker run --rm -u 503:20 -v <path-to-models-repo>/tools/docker:/tf -v <path-to-models-repo>/dockerfiles:/tf/dockerfiles imz-tf-tools python3 assembler.py --release dockerfiles --construct_dockerfiles
```

For single targets such as `bert-large-fp32-training` the model-builder adds an argument:

```
--only_tags_matching=.*bert-large-fp32-training$
```

### Building images

The model-builder command will build images by calling docker run on the imz-tf-tools container passing
in arguments to assembler.py. This internal call looks like the following:

```
docker run --rm -v <path-to-models-repo>/tools/docker:/tf -v /var/run/docker.sock:/var/run/docker.sock imz-tf-tools python3 assembler.py --arg _TAG_PREFIX=latest --arg http_proxy= --arg https_proxy= --arg TENSORFLOW_TAG=latest --arg PACKAGE_DIR=model_packages --arg MODEL_WORKSPACE=/workspace --repository model-zoo --release versioned --build_images --only_tags_matching=.*bert-large-fp32-training$ --quiet
```

For single targets such as `bert-large-fp32-training` the model-builder adds an argument:

```
--only_tags_matching=.*bert-large-fp32-training$
```

### Generating k8 deployments

The model-builder's generate-deployment subcommand will generate kubernetes deployments for that model using the model's k8 package. The subcommand will use env values from the model specification's runtime.env yaml array. The subcommand will look for runtime.env replacements under `${XDG_CONFIG_HOME:-$HOME/.config}/model-builder/specs/k8s`[^Ψ]. The first time `model-builder generate-deployment` is run, it will copy ./tools/docker/specs to `${XDG_CONFIG_HOME:-$HOME/.config}/model-builder/specs`. The user can then edit the k8 specification files to provide env values specific to their k8 deployment. For example the user would replace USER_ID with their [id](https://man7.org/linux/man-pages/man1/id.1.html). The subcommand algorithm is as follows:

[^Ψ]: If $HOME is a nfs mount point then you should chose an alternate directory eg: `XDG_CONFIG_HOME=/tmp/$USER/config model-builder generate-deployment ...`.

1. Read in all specification files under tools/docker/specs.
2. Copy tools/docker/specs to `${XDG_CONFIG_HOME:-$HOME/.config}/model-builder/specs` if the directory does not exist.
3. Read in all specification files under `${XDG_CONFIG_HOME:-$HOME/.config}/model-builder/specs`.
4. Replace any specifications in 1 with specifications from 2.
5. Extract the k8 package into a temporary directory.
6. For each env in the model's specification under runtime, call `kustomize cfg set <temp-dir> env.name env.value`.
7. Output the specific deployment under `deployments/<model>` by calling `kustomize build <temp-dir>`.

### Running test-suites

The model-builder's run-test-suit subcommand will generate tests for one or more subcommands where each subcommand can be given one or more release-groups and one or more models.
The syntax is shown below:

```
model-builder run-test-suite --command [build|generate-dockerfile|generate-documentation|generate-deployment|package] --release-group <release-group> <model> ...
```

The run-test-suite subcommand generates test cases using syntax compatible with [bats-core](https://github.com/bats-core/bats-core). After generation, it calls bats providing the generated script. Several examples with output are shown below

1. `model-builder run-test-suite --command generate-documentation --command generate-dockerfile --release-group versioned resnet50-fp32-inference resnet50-int8-inference`

```
 ✓ validate generate-documentation for resnet50-fp32-inference in release-group versioned creates quickstart/image_recognition/tensorflow/resnet50/inference/fp32/README.md
 ✓ validate generate-documentation for resnet50-int8-inference in release-group versioned creates quickstart/image_recognition/tensorflow/resnet50/inference/int8/README.md
 ✓ validate generate-dockerfile for resnet50-fp32-inference in release-group versioned creates intel-tf-image-recognition-resnet50-fp32-inference.Dockerfile
 ✓ validate generate-dockerfile for resnet50-int8-inference in release-group versioned creates intel-tf-image-recognition-resnet50-int8-inference.Dockerfile

4 tests, 0 failures
```

2. `model-builder run-test-suite -c package -f k8s`

```
 ✓ validate package for bert-large-fp32-training-k8s in framework k8s creates bert-large-fp32-training-k8s.tar.gz
 ✓ validate package for resnet50v1-5-fp32-inference-k8s in framework k8s creates resnet50v1-5-fp32-inference-k8s.tar.gz
 ✓ validate package for resnet50v1-5-fp32-training-k8s in framework k8s creates resnet50v1-5-fp32-training-k8s.tar.gz
 ✓ validate package for rfcn-fp32-inference-k8s in framework k8s creates rfcn-fp32-inference-k8s.tar.gz
 ✓ validate package for wide-deep-large-ds-fp32-training-k8s in framework k8s creates wide-deep-large-ds-fp32-training-k8s.tar.gz

5 tests, 0 failures
```

