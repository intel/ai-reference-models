# Scikit-learn with modin
Here you can find Docker file for Scikit-learn 

It's based on Scikit-learn conda package from Intel channel

To run the workflow try this:
```
$ docker run -it amr-registry.caas.intel.com/aipg-tf/model-zoo:intelpython3_core_census_workflow bash
$ python census_modin.py
```

Once script finishes running check for scores and ensure you get the following:
```
mean MSE ± deviation: 0.032564569 ± 0.000041799
mean COD ± deviation: 0.995367533 ± 0.000005869
```
