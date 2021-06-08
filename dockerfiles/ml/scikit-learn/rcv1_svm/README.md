To build the workflow container:

```
docker build -f Dockerfile -t intel/intel-optimized-ml:ubuntu-18.04-scikit-learn-rcv1-svm .
```

To run the workflow:
```
docker run -t intel/intel-optimized-ml:ubuntu-18.04-scikit-learn-rcv1-svm
```
