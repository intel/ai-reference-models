<!--- 20. Model package -->
## Model Package

The model package includes the scripts and libraries needed to
build and run <model name> <mode> using a docker container. Note that
this model container uses the Tensorflow ITEX GPU container as it's base,
and it requires the `model-zoo:tensorflow-itex-gpu` image to be built before
the model container is built.
```
<package dir>
├── build.sh
├── info.txt
├── licenses
│   ├── LICENSE
│   └── third_party
│       ├── Intel_Model_Zoo_NDA_TPP.txt
│       ├── Intel_Model_Zoo_v2.0_Container_tpps.txt
│       ├── Intel_Model_Zoo_v2.0_ML_Container_tpps.txt
│       ├── Intel_Model_Zoo_v2.3_PyTorch.txt
│       └── licenses.txt
├── model_packages
│   └── <package name>
├── README.md
├── run.sh
└── <package dir>.Dockerfile
```

