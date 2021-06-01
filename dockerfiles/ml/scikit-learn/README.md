# Scikit-learn with modin
Here you can find Docker file for Scikit-learn 

It's based on Scikit-learn conda package from Intel channel

To run the workflow try this:
```
docker run -it intel/intel-optimized-ml:scikit-learn-census-workflow bash
python census_modin.py
```

Once script finishes running check for scores and ensure you get the following:
```
Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) solvers for sklearn enabled: https://intelpython.github.io/daal4py/sklearn.html
Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) solvers for sklearn enabled: https://intelpython.github.io/daal4py/sklearn.html
UserWarning: User-defined function verification is still under development in Modin. The function provided is not verified.
mean MSE ± deviation: 0.032564569 ± 0.000041799
mean COD ± deviation: 0.995367533 ± 0.000005869
```
