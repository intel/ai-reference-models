## Model Package

The model package includes the Dockerfile and scripts needed to build and
run <model name> <mode> in a container.
```
<package dir>
├── README.md
├── build.sh
├── licenses
│   ├── LICENSE
│   └── third_party
│       ├── Intel_Model_Zoo_v2.4_Container_tpps.txt
│       ├── Intel_Model_Zoo_v2.4_ML_Container_tpps.txt
│       ├── Intel_Model_Zoo_v2.4_PyTorch.txt
│       └── licenses.txt
├── model_packages
│   └── <package name>
├──<package dir>.Dockerfile
└── run.sh
```
