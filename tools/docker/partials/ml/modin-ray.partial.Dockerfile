ENV MODIN_ENGINE ray

RUN python3 -m pip install \
    modin \
    ray

# Fix for this error: https://github.com/ray-project/ray/issues/6013
RUN sed -i.bak '/include_webui/ s/^#*/#/' ${CONDA_INSTALL_PATH}/lib/python3.7/site-packages/modin/engines/ray/utils.py
