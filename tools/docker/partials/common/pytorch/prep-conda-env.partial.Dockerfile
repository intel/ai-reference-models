# Prepare the Conda environment
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    chmod +x miniconda.sh && \
    ./miniconda.sh -b -p ~/conda && \
    rm ./miniconda.sh && \
    ~/conda/bin/conda create -yn pytorch python=3.7 && \
    export PATH=~/conda/bin/:${PATH} && \
    source activate pytorch && \
    pip install pip==21.0.1 && \
    conda config --add channels intel && \
    conda install -y ninja pyyaml setuptools cmake cffi typing intel-openmp psutil && \
    conda install -y mkl mkl-include numpy -c intel --no-update-deps

ENV PATH ~/conda/bin/:${PATH}
ENV LD_LIBRARY_PATH /lib64/:/usr/lib64/:/usr/local/lib64:/root/conda/envs/pytorch/lib:${LD_LIBRARY_PATH}
