# Note pycocotools has to be install after the other requirements
RUN pip install \
        Cython \
        contextlib2 \
        jupyter \
        lxml \
        matplotlib \
        numpy>=1.17.4 \
        'pillow>=9.3.0'

RUN pip install tqdm==4.43.0 \
    easydict==1.9 \
    scikit-image \
    pycocotools \
    opencv-python-headless
