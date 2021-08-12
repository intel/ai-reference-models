# Prepare the Conda environment
RUN wget --quiet https://repo.continuum.io/archive/Anaconda3-5.0.0-Linux-x86_64.sh -O anaconda3.sh && \
    chmod +x anaconda3.sh && \
    ./anaconda3.sh -b -p ~/anaconda3 && \
    rm ./anaconda3.sh && \
    ~/anaconda3/bin/conda create -yn pytorch python=3.7 && \
    export PATH=~/anaconda3/bin/:${PATH} && \
    source activate pytorch && \
    pip install pip==21.0.1 && \
    pip install sklearn onnx && \
    conda config --add channels intel && \
    conda install -y ninja pyyaml setuptools cmake cffi typing intel-openmp && \
    conda install -y mkl mkl-include numpy -c intel --no-update-deps

ENV PATH ~/anaconda3/bin/:${PATH}
ENV LD_LIBRARY_PATH /lib64/:/usr/lib64/:/usr/local/lib64:/root/anaconda3/envs/pytorch/lib:${LD_LIBRARY_PATH}
