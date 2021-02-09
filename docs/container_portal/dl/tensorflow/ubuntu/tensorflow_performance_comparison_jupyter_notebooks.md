
# Intel Optimizations for TensorFlow* Performance Comparison Jupyter* Notebooks

## Pull Command

```
docker pull intel/intel-optimized-tensorflow:2.4.0-ubuntu-20.04-jupyter-performance-comparison
```

## Description
This is a container with Jupyter* notebooks and pre-installed environments for analyzing the performance benefit from using Intel® Optimizations for TensorFlow* with Intel® oneAPI Deep Neural Network Library (Intel® oneDNN).
There are two different analysis types:
* For the "Stock vs. Intel® Optimizations for TensorFlow*" analysis type, users can understand the performance benefit between stock and Intel® Optimizations for TensorFlow* 
* For the "FP32 vs. BFloat16 vs. Int8"  analysis type, users can understand the performance benefit among different data types on Intel® Optimizations for TensorFlow*

| Analysis Type | Notebook | Notes|
| ------ | ------ | ------ |
|Stock vs. Intel® Optimizations for TensorFlow* | 1. [benchmark_perf_comparison](https://github.com/IntelAI/models/blob/master/docs/notebooks/perf_analysis/benchmark_perf_comparison.ipynb)  | Compare performance between stock and Intel® Optimizations for TensorFlow* among different models  |
|^| 2. [benchmark_perf_timeline_analysis](https://github.com/IntelAI/models/blob/master/docs/notebooks/perf_analysis/benchmark_perf_timeline_analysis.ipynb) | Analyze the performance benefit from Intel® oneDNN among different layers by using TensorFlow* Timeline |  
|FP32 vs. BFloat16 vs. Int8 | 1. [benchmark_data_types_perf_comparison](https://github.com/IntelAI/models/blob/master/docs/notebooks/perf_analysis/benchmark_data_types_perf_comparison.ipynb) | Compare Intel® Model Zoo benchmark performance among different data types on Intel® Optimizations for TensorFlow*  |
|^| 2.[benchmark_data_types_perf_timeline_analysis](https://github.com/IntelAI/models/blob/master/docs/notebooks/perf_analysis/benchmark_data_types_perf_timeline_analysis.ipynb) | Analyze the BFloat16/Int8 data type performance benefit from Intel® oneDNN among different layers by using TensorFlow* Timeline |  

#### How to Run the Notebooks
 
1. Launch the container with:
   ```
   docker run \
       -d \
       -p 8888:8888 \
       --env LISTEN_IP=0.0.0.0 \
       --privileged \
       intel/intel-optimized-tensorflow:2.4.0-ubuntu-20.04-performance-comparison-jupyter
   ```

   If your machine is behind a proxy, you will need to pass proxy arguments to the run command. For example:
   ```
   --env http_proxy="http://proxy.url:proxy_port" --env https_proxy="https://proxy.url:proxy_port"
   ```

2. Display the container logs with `docker logs <container_id>`, copy the jupyter service URL, and paste it into a browser window.

3. Click the 1st notebook file (`benchmark_perf_comparison.ipynb` or `benchmark_data_types_perf_comparison`) from an analysis type.

   > Note: For "Stock vs. Intel® Optimizations for TensorFlow*" analysis type, please change your Jupyter* notebook kernel to either "stock-tensorflow" or "intel-tensorflow"
    
   > Note: For "FP32 vs. BFloat16 vs. Int8" analysis type, please select "intel-tensorflow" as your Jupyter* notebook kernel.

4. Run through every cell of the notebook one by one

   > NOTE: For "Stock vs. Intel® Optimizations for TensorFlow*" analysis type, in order to compare between stock and Intel® Optimizations for TensorFlow* results, users need to run all cells before the comparison section with both stock-tensorflow and intel-tensorflow kernels. 

5. Click the 2nd notebook file (`benchmark_perf_timeline_analysis.ipynb` or `benchmark_data_types_perf_timeline_analysis`) from an analysis type.
6. Run through every cell of the notebook one by one to get the analysis result.

   > NOTE: There is no requirement for the Jupyter* kernel when users run the 2nd notebook to analyze performance in detail.

## Documentation and Sources

- [Docker Repo](https://hub.docker.com/r/intel/intel-optimized-tensorflow)
- [Main Github](https://github.com/IntelAI/models)
- [Readme](https://github.com/IntelAI/models/blob/master/docs/notebooks/perf_analysis/README.md)
- [Release Notes](https://github.com/IntelAI/models/releases)
- [Get Started Guide](https://github.com/IntelAI/models/blob/master/docs/notebooks/perf_analysis/README.md)
- [Dockerfile](https://github.com/IntelAI/models/tree/master/dockerfiles/notebook_containers)
- [Report Issue](https://github.com/IntelAI/models/issues)

## License Agreement
LEGAL NOTICE: By accessing, downloading or using this software and any required dependent software (the “Software Package”), 
you agree to the terms and conditions of the software license agreements for the Software Package, which may also include notices, 
disclaimers, or license terms for third party software included with the Software Package. 
Please refer to the [license file](http://github.com/IntelAI/models/tree/master/LICENSE) for additional details.

## Metadata
This is for internal use on the Intel® oneContainer Portal.

- SEO Keyword: TensorFlow performance Jupyter notebooks
- Search/Browser Title: TensorFlow* Performance Comparison Jupyter* Notebooks
- Search Description: Use TensorFlow* performance Jupyter* notebooks to analyze the performance benefit of Intel® Optimizations for TensorFlow*.
- Short Title: TensorFlow* Performance Jupyter* Notebooks
- Short Description: Use Jupyter* notebooks to analyze the performance benefit of Intel® Optimizations for TensorFlow*.
- Intel Keywords: TensorFlow, Jupyter, oneDNN, Deep Learning, Performance Analysis, Performance
- OS Tags (choose at least one):
  - [ ] Linux* / CentOS Linux Family*
  - [ ] Linux*
  - [ ] Debian Linux*
  - [X] Ubuntu*
  - [ ] Microsoft Windows*
  - [ ] Linux* / Red Hat Linux Family
  - [ ] SUSE Linux Family* 
- Platform Tags (can choose multiple):
  - [X] CPU
  - [ ] GPU/GPGPU
  - [ ] Intel® FPGA Technologies
  - [ ] Accelerators
  - [ ] Server
- Use Case Tags (can choose multiple):
  - [X] AI Inference
  - [ ] AI Training
  - [ ] Natural Language Processing (NLP)
  - [ ] Deep Reinforcement Learning (DRL)
  - [ ] Smart Video
  - [X] Image Recognition
  - [ ] Object Detection
  - [ ] Recommendation Engine
  - [ ] Analytics
  - [ ] Media Processing
  - [ ] Code Modernization
  - [ ] Platform Analysis
  - [X] Tuning and Performance Monitoring
  - [ ] Porting
  - [ ] Big Data
  - [ ] Media Delivery
  - [ ] Cloud Gaming & Graphics
  - [ ] Immersive Media
  - [ ] Cloud Computing
  - [ ] Edge Computing
  - [ ] Audio
  - [ ] Storage & Memory
  - [ ] Visual Computing 
- App Domain Tags (can choose multiple): 
  - [X] Artificial Intelligence
  - [ ] Rendering
  - [ ] Computer Vision
  - [ ] Software and Driver
  - [ ] Internet of Things (IoT)
  - [ ] Data Center
  - [ ] High Performance Computing (HPC)
  - [ ] Cloud & Edge Computing
  - [ ] Media Analytics
