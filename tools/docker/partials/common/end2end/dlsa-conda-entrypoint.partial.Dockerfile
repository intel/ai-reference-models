SHELL [ "/bin/bash", "-c" ]

RUN echo "source activate dlsa" >> ~/.bashrc

ENTRYPOINT [ "conda", "run", "--no-capture-output", "-n", "dlsa" ]
