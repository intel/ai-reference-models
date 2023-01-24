<!--- 40. Quick Start Scripts -->
## Quick Start Scripts

| Script name | Description |
|-------------|-------------|
| [`multi_instance_training.sh`](/quickstart/image_recognition/tensorflow/resnet50v1_5/training/cpu/multi_instance_training.sh) |	Uses mpirun to execute 1 processes with 1 process per socket with a batch size of 1024 for the specified precision (fp32 or bfloat16 or bfloat32). Checkpoint files and logs for each instance are saved to the output directory.|
