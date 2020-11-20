#!/usr/bin/env bash
die() {
    echo $@
    exit 1
}

python -c '
try:
    import torch
    print(True)
except:
    print(False)
'
pytorch_available=$?

if [[ $pytorch_available -eq 0 ]]; then
       echo "PASS: Pytorch is available"
else
       die "FAIL: Could not import pytorch"
fi
