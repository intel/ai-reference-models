<!--- 20. Model package -->
## Model Package

The model package includes the scripts and libraries needed to
build and run <model name> <mode> using a docker container. Note that
this model container uses the PyTorch IPEX GPU container as it's base,
and it requires the `model-zoo:pytorch-ipex-gpu` image to be built before
the model container is built.
```
<package dir>
├── build.sh
├── info.txt
├── licenses
│   ├── LICENSE
│   └── third_party
├── model_packages
│   └── <package name>
├── <package dir>.Dockerfile
├── README.md
└── run.sh
```
