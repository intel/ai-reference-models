#!/usr/bin/env bash

die() {
    echo $@
    exit 1
}

python -c '
try:
    import torch
    import torch_ccl
    print(True)
except:
    print(False)
'
torchccl_available=$?

if [[ $torchccl_available -eq 0 ]]; then
       echo "PASS: Torch-CCL is available"
else
       die "FAIL: Could not import torch_ccl"
fi
