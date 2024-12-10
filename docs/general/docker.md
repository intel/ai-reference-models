# Docker

The Model Zoo for Intel® Architecture has instructions for running models
using [docker containers](https://www.docker.com/resources/what-container)
and on bare metal. Using containers helps to provide a known and tested
environment for running your model, but it can also bring challenges if
you are not familiar with how docker works. Below are some hints for
troubleshooting issues that are commonly encountered by new users.

# Permission denied when running docker

When running docker as a non-root user, you may get a permission denied error
message similar to this:
```
$ docker run -it intel/intel-optimized-tensorflow /bin/bash
docker: Got permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: Post http://%2Fvar%2Frun%2Fdocker.sock/v1.40/containers/create: dial unix /var/run/docker.sock: connect: permission denied.
See 'docker run --help'.
```

There are a couple of common ways to resolve this issue:
* Add your user to the docker group:
  ```
  sudo usermod -a -G docker $USER
  ```
  After running this command, logout and restart your terminal and then retry the docker command.
* Run docker using `sudo`. For example:
  ```
  sudo docker run -it intel/intel-optimized-tensorflow /bin/bash
  ```

## Mounting volumes

Mounting a volume allows the container to read and write files from your
local system. The model zoo often uses volume mounts to give the model access
to datasets and pretrained models, and to provide a location for the scripts
to write log files and checkpoint files that are generated during training.

If you get a "permission denied" error from your volume mount, ensure that
the directory you are mounting is on your local file system (not NFS) and
that docker (running as root) has read/write access to the directory.

If you really need to use an NFS directory as your volume mount, see the
[docker volume create](https://docs.docker.com/engine/reference/commandline/volume_create/)
command where you can use the `--opt` flag to specify the type as nfs.
