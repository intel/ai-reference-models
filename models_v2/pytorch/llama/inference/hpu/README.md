<!--- 0. Title -->
# PyTorch LLaMA2 and LLaMA38B inference using Optimum-Habana and vLLM on Gaudi (generation)

<!-- 10. Description -->
## Description

This document has instructions for running Llama2 and Llama3 inference (generation) using Optimum-Habana and vLLM on Gaudi.

## Bare Metal
### General setup
Please make sure to follow [Driver Installation](https://docs.habana.ai/en/latest/Installation_Guide/Driver_Installation.html) to install Gaudi driver on the system.
To use dockerfile provided for the sample, please follow [Docker Installation](https://docs.habana.ai/en/latest/Installation_Guide/Additional_Installation/Docker_Installation.html) to setup habana runtime for Docker images.

### Model Specific Setup
    For both vLLM and Optimum-habana, Gaudi-tutorials is required.
    ```
    git clone https://github.com/HabanaAI/Gaudi-tutorials.git
    ```
    Please also follow CONTAINER.md to run docker instance. 
    
# Inference

## Optimum-Habana
Benchmark script will run all the models with different input len, output len and batch size and generate a report to compare all published numbers in [Gaudi Model Performance](https://www.intel.com/content/www/us/en/developer/platform/gaudi/model-performance.html).  

Different json file are provided for different Gaudi Software version like 1.19 and 1.20 on Gaudi3.
To do benchmarking on a machine with 8 Gaudi3 cards, just run the below command inside the docker instance under /workspace/optimum-habana/examples/text-generation. 
```bash
cd /workspace/optimum-habana/examples/text-generation
python3 Benchmark.py
```

## vLLM

## Output

### Optimum-Habana

![image](https://github.com/user-attachments/assets/db44e6b5-4be2-4559-a5d2-f369e35cf4ea)

### vLLM

<!--- 80. License -->
## License
[LICENSE](https://github.com/IntelAI/models/blob/master/LICENSE)
