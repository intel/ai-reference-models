ARG PACKAGE_NAME
ARG MODEL_WORKSPACE
    
RUN pip install -r ${MODEL_WORKSPACE}/${PACKAGE_NAME}/models/language_modeling/pytorch/bert_large/training/gpu/requirements.txt

RUN cd ${MODEL_WORKSPACE}/${PACKAGE_NAME}/models/language_modeling/pytorch/bert_large/training/gpu/data/ && \
    wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt && \
    mv bert-base-uncased-vocab.txt vocab. && \
    cd -
