
## Tips to Reduce the Processing Time

The ImageNet 2012 validation contains 50000 images. Therefore, the LPOT will take hours to quantize and evaluate the models.  

There are several tips to reduce the whole data processing time:

### 1. Reduce the number of images for calibration.


The number of images for calibration is defined as 'sampling_size'. 
In the example below, only 50 images are used for calibration.

```
quantization:
  calibration:
    sampling_size: 50 
```  

Users could also set multiple values like example below:

```
quantization:
  calibration:
    sampling_size: 50, 100, 200, 400, 800

```

### 2. Reduce the number of images for evaluation.
   
The number of images for evaluation should not be less than the batch size.

   
### 3. Set timeout in YAML file:

It will force LPOT to stop when it reaches the timeout time.

In below example, LPOT will be stopped after 1000 secs.

```
tuning:
  ...
  exit_policy:
    timeout: 1000  
```
