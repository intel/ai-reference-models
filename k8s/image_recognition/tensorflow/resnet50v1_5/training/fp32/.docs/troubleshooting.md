<!--- 71. TroubleShooting -->
## TroubleShooting

- Pod doesn't start. Status is ErrImagePull.<br/>
  Docker recently implemented rate limits.<br/>
  See this [note](https://thenewstack.io/docker-hub-limits-what-they-are-and-how-to-route-around-them/) about rate limits and work-arounds.

- Argo workflow steps do not execute.<br/>
  Error from `argo get <workflow>` is 'failed to save outputs: Failed to establish pod watch: timed out waiting for the condition'.<br/>
  See this argo [issue](https://github.com/argoproj/argo/issues/4186). This is due to the workflow running as non-root.<br/>
  Devops will need to change the workflow-executor to k8sapi as described [here](https://github.com/argoproj/argo/blob/master/docs/workflow-executors.md).

- MpiOperator can't create workers. Error is '/bin/sh: /etc/hosts: Permission denied'. This is due to a bug in mpi-operator in the 'latest' container image
  when the workers run as non-root. See this [issue](https://github.com/kubeflow/mpi-operator/issues/288).<br/>
  Use the container images: mpioperator/mpi-operator:v02.3 and mpioperator/kubectl-delivery:v0.2.3.

