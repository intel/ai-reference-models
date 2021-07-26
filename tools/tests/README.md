
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
   Currently, the automated tests cover model-builder subcommands: generate-dockerfile, generate-documentation, generate-deployment, build, and package.
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

### K8s Model deployment tests:

The command `model-builder run-test-suite -c generate-deployment -f k8s <target-k8s-model>`
generates and tests kubernetes deployments for the target model using the model's k8 package such as `bert-large-fp32-training-k8s`. The supported K8s models are listed in `models/k8s`.

>Note: 
> * If $HOME is a nfs mount point then you should chose an alternate directory eg: `XDG_CONFIG_HOME=/tmp/$USER/config model-builder run-test-suite generate-deployment ...`.
> * The user can edit the [k8 specification files](models/tools/docker/specs/k8s) to provide env values specific to their k8 deployment. For example the user would replace USER_ID with their [id](https://man7.org/linux/man-pages/man1/id.1.html).

`generate-deployment` in `model-builder-test-suite-generator` aggregates a list of test snippets based on the model specification's `runtime.tests` and append them to `model-builder.bats`

Example for the bert-large-fp32-training-k8s specification's `runtime.tests`:
```
        tests:
          - uri: k8s/language_modeling/tensorflow/bert_large/training/fp32/.tests/deployments-yml-exist.sh
            args:
              - name: ''
                value: ''
          - uri: k8s/language_modeling/tensorflow/bert_large/training/fp32/.tests/namespace-exists.sh
            args:
              - name: USER_NAME
                value: $USER_NAME
          - uri: k8s/language_modeling/tensorflow/bert_large/training/fp32/.tests/single_node_deployment.sh
            args:
              - name: ''
                value: ''
```
`uri` refers to each test snippet location in the `.tests` such as `models/k8s/language_modeling/tensorflow/bert_large/training/fp32/.tests`.

The test output looks like:
```
 ✓ validate package for bert-large-fp32-training-k8s in framework k8s creates model package file 
 ✓ validate generate-deployment for bert-large-fp32-training-k8s in framework k8s creates deployment files under deployments/bert-large-fp32-training-k8s/ 
 ✓ verify namespace <USER_NAME> exists 
 ✓ kubectl apply of single-node bert-large-fp32-training-k8s 
 ✓ single-node deployment of bert-large-fp32-training-k8s creates 1 pod and it is running 
 ✓ kubectl delete of single-node bert-large-fp32-training-k8s 

6 tests, 0 failures
```