# mpi-operator

Based on [mpi-operator](https://github.com/kubeflow/mpi-operator/tree/master/deploy/v1).
Both images within the Deployment use tag v0.2.3.

The latest tag for mpi-operator has problems creating workers. The error is '/bin/sh: /etc/hosts: Permission denied'. 
This is due to a bug in mpi-operator in the 'latest' container image when the workers run as non-root. 
See this [issue](https://github.com/kubeflow/mpi-operator/issues/288).<br/>

## Install

> kubectl apply -f mpi-operator.yaml

## Delete

> kubectl delete -f mpi-operator.yaml

