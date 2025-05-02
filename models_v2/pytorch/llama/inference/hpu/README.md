<!--- 0. Title -->
# PyTorch LLaMA2 and LLaMA3 inference using Optimum-Habana and vLLM on Gaudi (generation)

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
    
Please also follow  [CONTAINER.md](./CONTAINER.md) to run docker instance. 
    
# Inference

## Optimum-Habana
Benchmark script will run all the models with different input len, output len and batch size and generate a report to compare all published numbers in [Gaudi Model Performance](https://www.intel.com/content/www/us/en/developer/platform/gaudi/model-performance.html).   

Different json file are provided for different Gaudi Software version like 1.19 and 1.20 on Gaudi3.  
To do benchmarking on a machine with 8 Gaudi3 cards, just run the below command inside the docker instance under /workspace/optimum-habana/examples/text-generation.  
```bash
cd /workspace/optimum-habana/examples/text-generation
./run_model.sh
```

## vLLM

There are different folders for different models, but all models share the same docker image.
please refer the Gaudi [vLLM Tutorials README](https://github.com/HabanaAI/Gaudi-tutorials/tree/main/PyTorch/vLLM_Tutorials/Benchmarking_on_vLLM/Online_Static) for more details.  

Here are instructions to run llama-3.1-70b-instruct model on gaudi3 with 1.20 release.  
1. brower into model folder: 
```bash
cd PyTorch/vLLM_Tutorials/Benchmarking_on_vLLM/Online_Static/llama-3.1-70b-instruct_gaudi3_1.20_contextlen-2k
```

2. Edit the docker_envfile.env and enter your HF_TOKEN in the placeholder variable. Optionally, you can set which cards to use by changing HABANA_VISIBLE_DEVICES.

3. Run the vllm server using run.sh
```bash
chmod +x run.sh
./run.sh
```

   Wait ~15 minutes or more for the server to start and warmup.   
   Ignore the pulsecheck   | No successful response. HTTP status code: 000. Retrying in 5 seconds... messages in the meantime.    
   When server is finally ready for serving, it will say Application Startup Complete. INFO:     Uvicorn running on http://0.0.0.0.:8000  

4. run simple test
First, using the runing gaudi-benchmark docker instance
```bash
docker exec gaudi-benchmark /bin/bash
```
Second, run below commands to do simple test
```bash
cd /workdir/
./perftest.sh
```

## Output

### Optimum-Habana

![image](https://github.com/user-attachments/assets/db44e6b5-4be2-4559-a5d2-f369e35cf4ea)


### vLLM

![image](https://github.com/user-attachments/assets/02a2ea48-7c27-461a-8dff-9529f27b304c)




<!--- 80. License -->
## License
[LICENSE](https://github.com/IntelAI/models/blob/master/LICENSE)
