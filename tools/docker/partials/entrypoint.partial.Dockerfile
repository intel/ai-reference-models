
ENV USER_ID=0
ENV USER_NAME=root
ENV GROUP_ID=0
ENV GROUP_NAME=root

#TODO this needs to be approved by SDL
#See https://gitlab.devtools.intel.com/TensorFlow/QA/cje-tf/-/merge_requests/685#note_4718598
RUN apt-get update && apt-get install -y gosu
RUN echo '#!/bin/bash\n\
USER_ID=$USER_ID\n\
USER_NAME=$USER_NAME\n\
GROUP_ID=$GROUP_ID\n\
GROUP_NAME=$GROUP_NAME\n\
if [[ $GROUP_NAME != root ]]; then\n\
  groupadd -r -g $GROUP_ID $GROUP_NAME\n\
fi\n\
if [[ $USER_NAME != root ]]; then\n\
  useradd --no-log-init -r -u $USER_ID -g $GROUP_NAME -s /bin/bash -M $USER_NAME\n\
fi\n\
exec /usr/sbin/gosu $USER_NAME:$GROUP_NAME "$@"\n '\
>> /tmp/entrypoint.sh
RUN chmod u+x,g+x /tmp/entrypoint.sh
ENTRYPOINT ["/tmp/entrypoint.sh"]
