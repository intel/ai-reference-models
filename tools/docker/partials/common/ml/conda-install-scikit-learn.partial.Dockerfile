RUN conda install -y -q \
        daal4py \
        scikit-learn-intelex \
        threadpoolctl && \
    conda clean -y --all

ENV PYTHONSTARTUP=${HOME}/.patch_sklearn.py

RUN echo \
'from sklearnex import patch_sklearn\n\
from sklearnex import unpatch_sklearn\n\
patch_sklearn()\n\
print("To disable Intel(R) Extension for Scikit-learn*, you can run: unpatch_sklearn()")\n' \
>> ${PYTHONSTARTUP}
