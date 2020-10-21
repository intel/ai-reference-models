# Kubernetes Dev Ops

## MPI Operator Deployment

The following instructions explain how to deploy the MPI Operator to a Kubernetes
cluster. This deployment needs to be done by a devops user with Cluster RBAC permissions.
The steps below explain how to deploy the mpi-operator on your cluster using version `v0.2.3`.

1. Download the `mpi-operator.yaml` file
   ```
   curl -o mpi-operator.yaml https://github.com/kubeflow/mpi-operator/blob/v0.2.3/deploy/v1/mpi-operator.yaml
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
the cluster. See [Argo's Getting Started Guide](https://github.com/argoproj/argo/blob/stable/docs/getting-started.md)
for installation instructions.

Running one of the [example workflows](https://github.com/argoproj/argo/blob/stable/docs/getting-started.md#4-run-sample-workflows)
and verifying that it succeeds is a good test to verify that the argo
installation is working.
