## Container Package

The container package includes wheels for PyTorch and the Intel PyTorch
Extension (IPEX), a Dockerfile, and a script to build the container.

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
├── pytorch-ipex-spr.Dockerfile
└── whls
    ├── torch-1.10.0a0+git6f40371-cp37-cp37m-linux_x86_64.whl
    └── torch_ipex-1.1.0-cp37-cp37m-linux_x86_64.whl
```
