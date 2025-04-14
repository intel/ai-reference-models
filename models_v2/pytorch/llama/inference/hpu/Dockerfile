FROM vault.habana.ai/gaudi-docker/1.20.0/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest

# Need node to build doc HTML. Taken from https://stackoverflow.com/a/67491580
RUN apt-get update && apt-get install -y \
    software-properties-common \
    npm
RUN npm install n -g && \
    n latest

RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --upgrade-strategy eager optimum[habana]
RUN python3 -m pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.20.0
RUN mkdir -p /workspace
WORKDIR /workspace

## Install vllm-fork inside the container
RUN git clone https://github.com/HabanaAI/vllm-fork.git && \
    cd vllm-fork && \
    git checkout v0.6.6.post1+Gaudi-1.20.0 && \
    git checkout 6af2f67 && \
    pip install -v -e . && \
    pip install datasets

RUN ln -sf /usr/bin/python3 /usr/bin/python

## Install optimum-habana inside the container
RUN git clone https://github.com/huggingface/optimum-habana && cd optimum-habana && git checkout v1.16.0

WORKDIR /workspace/optimum-habana/examples/text-generation
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install -r requirements_lm_eval.txt
COPY ./Gaudi-tutorials/PyTorch/Hugging_Face_pipelines/Benchmarking_on_Optimum-habana_with_fp8/. .
COPY ./Gaudi-tutorials/PyTorch/Hugging_Face_pipelines/Benchmarking_on_Optimum-habana_with_fp8/Gaudi_1-20.json Gaudi.json
COPY ./Gaudi-tutorials/PyTorch/Hugging_Face_pipelines/Benchmarking_on_Optimum-habana_with_fp8/HQT_1-20.zip HQT.zip
RUN python3 -m pip install -r requirements_bm.txt

WORKDIR /workspace
