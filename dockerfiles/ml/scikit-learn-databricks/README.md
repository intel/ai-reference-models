# Scikit-learn daal4py and TensorFlow
Here you can find Docker file for Scikit-learn

It's based on Scikit-learn and TensorFlow conda packages from Intel channel

To build the container try this:
```
docker build -f scikit-learn-databricks.Dockerfile . -t intel/intel-optimized-ml:tf-2.4.0-scikit-learn
```

To run the workflow try this:
```
docker run -it intel/intel-optimized-ml:tf-2.4.0-scikit-learn bash
```
