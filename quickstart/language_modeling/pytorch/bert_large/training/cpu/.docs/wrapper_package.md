## Model Package

The model package includes the Dockerfile and scripts needed to build and
run <model name> <mode> in a container.
```
<package dir>
├── README.md
├── build.sh
├── input_preprocessing
│   ├── chop_hdf5_files.py
│   ├── create_pretraining_data.py
│   ├── create_pretraining_data_wrapper.sh
│   ├── prallel_create_hdf5.sh
│   └── tokenization.py
├── licenses
│   ├── LICENSE
│   └── third_party
├── model_packages
│   └── <package name>
├── <package dir>.Dockerfile
└── run.sh
```
