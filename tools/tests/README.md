
# Tests for models 

## Moving towards a [GitOps](https://about.gitlab.com/topics/gitops/) paradigm

1. Model testing will use the git repository as the source of truth
1. Model tests will be declarative, based on model specifications versioned in git under docker/specs
1. Model tests will be generated as test artifacts from these model specifications
1. Model tests and test outputs will be compliant with the [Test Anything Protocol (TAP)](http://testanything.org/).


## CICD

As part of Jenkins CICD the following steps will be included for MRs and general regression testing:

1. Call `model-builder run-test-suite`

   This subcommand will call model-builder-test-suite-generator which will auto-discover all models by calling `model-builder models`.
   This call returns all models declared under `tools/docker/specs`. Each model's manifest includes information
   used to generate a Dockerfile, build a container and package the model. The generator will enumerate over the set of models
   and generate a set of tests for each model. These tests validate model-builder subcommands and are appended to model-builder.bats.
   Currently, the automated tests cover model-builder subcommands: generate-dockerfile, generate-documentation, build, and package.
   Additional tests related to model creation (model-builder init-spec) and model deployment (docker, k8s) will be added over time.

   These tests look like:

```
@test "validate generate-dockerfile for bert-large-fp32-training creates intel-tf-language-modeling-bert-large-fp32-training.Dockerfile" {
    run model-builder -q generate-dockerfile bert-large-fp32-training
    (( $status == 0 ))
    [[ ${lines[@]} =~ intel-tf-language-modeling-bert-large-fp32-training.Dockerfile ]]
    (( $(last_modified dockerfiles/model_containers/intel-tf-language-modeling-bert-large-fp32-training.Dockerfile) <= 50 ))
  }
@test "validate generate-dockerfile for image-recognition creates intel-tf-image-recognition.Dockerfile" {
    run model-builder -q generate-dockerfile image-recognition
    (( $status == 0 ))
    [[ ${lines[@]} =~ intel-tf-image-recognition.Dockerfile ]]
    (( $(last_modified dockerfiles/intel-tf-image-recognition.Dockerfile) <= 50 ))
  }
```

  The test output (compliant with [Test Anything Protocol (TAP)](http://testanything.org/)) looks like:

```
 ✓ validate 'model-builder commands'
 ✓ validate 'model-builder models'
 ✓ validate generate-dockerfile for bert-large-fp32-training creates intel-tf-language-modeling-bert-large-fp32-training.Dockerfile
 ✓ validate generate-dockerfile for image-recognition creates intel-tf-image-recognition.Dockerfile
 ✓ validate generate-dockerfile for language-modeling creates intel-tf-language-modeling.Dockerfile
 ✓ validate generate-dockerfile for object-detection creates intel-tf-object-detection.Dockerfile
 ✓ validate generate-dockerfile for preprocess-coco-val creates intel-tf-object-detection-preprocess-coco-val.Dockerfile
```
