ARG PACKAGE_NAME
ARG MODEL_WORKSPACE

RUN cd ${MODEL_WORKSPACE}/${PACKAGE_NAME}/models/language_modeling/pytorch/bert_large/inference/gpu && \
    pip install -r requirements.txt 
    
RUN cd -
