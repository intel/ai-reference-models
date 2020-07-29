# Kubernetes Mlops

## Using NFS as storage

Configuring the quickstart to use NFS as storage requires specifying these values in the mlops.env within the tree show below:

```
quickstart
└── common
    └── k8s
        └── mlops
            ├── base
            │   └──  mlops.env
            ├── multi-node
            └── single-node
```

The NFS related values within mlops.env are shown below:

```
NFS_PATH=/exported_users
NFS_MOUNT_PATH=/home
NFS_SERVER=0.0.0.0
```

They should reflect values specific to your NFS implementation. NFS_PATH and NFS_SERVER are typically found in /etc/mtab 
and are NFS server values. NFS_MOUNT_PATH is a nfs client option indicating where the exported file system is mounted at.
