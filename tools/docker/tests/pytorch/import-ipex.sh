#!/usr/bin/env bash

die() {
    echo $@
    exit 1
}

python -c '
try:
    import intel_pytorch_extension
    print(True)
except:
    print(False)
'
ipex_available=$?

if [[ $ipex_available -eq 0 ]]; then
       echo "PASS: IPEX is available"
else
       die "FAIL: Could not import IPEX"
fi
