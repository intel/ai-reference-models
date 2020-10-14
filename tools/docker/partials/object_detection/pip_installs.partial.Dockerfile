# Note pycocotools has to be install after the other requirements
RUN pip install numpy==1.17.4 Cython contextlib2 pillow>=7.1.0 lxml jupyter matplotlib && \
    pip install pycocotools

