To build the workflow container:

```
docker build -f Dockerfile -t amr-registry.caas.intel.com/aipg-tf/model-zoo:ubuntu-18.04-scikit-learn-rcv1-svm .
```

To run the workflow:
```
docker run -t amr-registry.caas.intel.com/aipg-tf/model-zoo:ubuntu-18.04-scikit-learn-rcv1-svm
```
