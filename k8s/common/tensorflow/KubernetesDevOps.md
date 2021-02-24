# Kubernetes Dev Ops

## MPI Operator Deployment

The following instructions explain how to deploy the [MPI Operator](https://github.com/kubeflow/mpi-operator/)
to a Kubernetes cluster. This deployment needs to be done by a devops user with Cluster RBAC permissions.

We have a [copy of the `mpi-operator.yaml`](/tools/k8s/devops/operators/mpi-operator/mpi-operator.yaml)
that is compatible with our multi-node Kubernetes packages. There are instructions on how to deploy this
mpi-operator in this [README](/tools/k8s/devops/operators/mpi-operator/README.md).

If you prefer to install it from the [MPI Operator repo](https://github.com/kubeflow/mpi-operator/),
follow the steps below that explain how to deploy the mpi-operator on your cluster using version `v0.2.3`.

1. Download the v1alpha2 `mpi-operator.yaml` file from the v0.2.3 tag
   ```
   curl -o mpi-operator.yaml https://raw.githubusercontent.com/kubeflow/mpi-operator/v0.2.3/deploy/v1alpha2/mpi-operator.yaml
   ```

2. Change line #189 from `image: mpioperator/mpi-operator:latest` to `image: mpioperator/mpi-operator:v0.2.3`

3. Deploy the operator
   ```
   kubectl apply -f mpi-operator.yaml
   ```

For more information on deploying the mpi-operator to the k8s cluster, see the
[documentation in the mpi-operator repo](https://github.com/kubeflow/mpi-operator#mpi-operator).

## Argo Deployment

Prior to running Argo workflows, the controller needs to be deployed on
the cluster. See [Argo's Quick Start Guide](https://github.com/argoproj/argo-workflows/blob/stable/docs/quick-start.md)
for installation instructions.

Running the [hello world workflow](https://github.com/argoproj/argo-workflows/blob/stable/docs/quick-start.md)
and verifying that it succeeds is a good test to verify that the argo
installation is working. Using the Argo CLI is optional. The argo yaml
can be deployed using `kubectl create`.
