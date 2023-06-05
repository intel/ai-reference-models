<!-- 50. Launch benchmark instructions -->
Once your environment is setup, navigate to the `benchmarks` directory of
the model zoo and set environment variables for the directory, checkpoint
directory, and an output directory where log files will be written.
```
cd benchmarks

export OUTPUT_DIR=<directory where checkpoints and log files will be saved>
```

<model name> <mode> can be run for multiple tasks

* **To run GLUE pre-training** use the following command with the `train_option=GLUE`. The
  `CHECKPOINT_DIR` should point to the location where you've downloaded the gpt-j model if 
  not using the default from Huggingface
  ```
OMP_NUM_THREADS=4  KMP_AFFINITY=granularity=fine,verbose,compact,1,0  numactl -C 0-56 python ./benchmarks/launch_benchmark.py \
        --framework tensorflow \
        --model-name "gpt_j_6B" \
        --mode training \
        --precision fp32 \
        --batch-size 8 \
        --num-intra-threads 56 \
        --num-inter-threads 2 \
        --verbose  \
        -- train_option="GLUE" \
           task_name="WNLI" \
           pad_to_max_length=True \
           cache_dir=./cache \
           learning_rate=2e-5 \
           output_dir=./output \
           do_train=True \
           do_eval=True \
           do_predict=True \
           warmup_steps=5 \
           steps=20 \
           num_train_epochs=1

  ```

