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

# mpi-operator-namespaced

`mpi-operator-namespaced.yaml` is based on `mpi-operator.yaml` & helps address a different use case than the vanilla mpi-operator.

This approach was born out of an issue we encountered with multiple mpi-operators in the same cluster.

## Problem Statement:

Consider two mpi-operators launched within the same cluster but in different namespaces.
When an MPIJob is launched by any user on that cluster, both mpi-operators would listen 
to any new MPIJob that has launched anywhere on the cluster.
Because of this, an issue with mpi-operator leader election occurs, causing none of the mpi-operators to 
take up the newly created MPIJob and hence failing to create any pods.

## Solution:

When mpi-operator launches the operator it starts with some default settings, 
- it automatically scopes the operator to `mpi-operator` namespace. 
- it also locks the operator to `mpi-operator` namespace.

To override these defaults, `mpi-operator` provides two run-time container args that allow 
each mpi-operator to be scoped to a particular namespace as opposed to a Cluster scope.

```
--namespace=<your-namespace>
--lock-namespace=<your-namespace>
```

Adding these to the dynamic container args section would run the mpi-operator docker image with the namespace you specify.
This will enable the mpi-operator to listen to the MPIJobs that originate only in the specified namespace and not cluster-wide.

This solution eliminates the issue of leader election in a cluster scoped setting.

## Uses:

The recommended use of this configuration would be when the user would like to scope the 
operator to a particular namespace & avoid conflicts with any other actively running
mpi-operators on the cluster.
