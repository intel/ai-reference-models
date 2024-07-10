# GEMMA Inference

Text generation with GEMMA using KerasNLP, a collection of natural language processing (NLP) models implemented in Keras and runnable on JAX, PyTorch, and TensorFlow.

## Model Information

| **Use Case** | **Framework** | **Model Repo** |          **Branch/Commit/Tag**           |  **Optional Patch** |
|:---:| :---: | :---: |:----------------------------------------:| :---: |
|  Inference   |  Keras   | [gemma](https://ai.google.dev/gemma/docs/get_started) | - |  - |

## Prerequisites

The model checkpoints are available through Kaggle at http://kaggle.com/models/google/gemma. Select one of the keras model variations, click the ⤓ button to download the model archive, then extract the contents to a local directory. The archive contains the model weights, the tokenizer and other necessary files to load the model. An example of what the extracted archive of `gemma_2b_en` keras model looks like:

```
assets
config.json
metadata.json
model.weights.h5    # Model weights
tokenizer.json      # Tokenizer
```
Once you download the file `archive.tar.gz`, untar the file and point the unzipped directory to `MODEL_PATH`.

## Inference
1. `git clone https://github.com/IntelAI/models.git`
2. `cd models/models_v2/jax/gemma/inference/cpu`
3. Create virtual environment `venv` and activate it:
    ```
    python3 -m venv venv
    . ./venv/bin/activate
    ```
4. Setup required environment variables for setup
    |   **Environment Variable**    |    **Purpose**    |                   **export command**                    |
    |:-------------------------------------------------------:| :-------------------------------------------------------:| :---: |
    |   <div style="text-align: left"> **JAX_NIGHTLY (optional)** </div>   |   Set to 1 to install the nightly release of JAX. If not set to 1, defaults to the public release of JAX     |     <div style="text-align: left"> `export JAX_NIGHTLY=1` </div> |
5. Run `setup.sh`
    ```
    ./setup.sh
    ```
6. Setup required environment variables for running the model

    |   **Environment Variable**    |    **Purpose**    |                   **export command**                    |
    |:-------------------------------------------------------:| :-------------------------------------------------------:| :---: |
    |   <div style="text-align: left"> **PRECISION** </div>   |     Determine the precision for inference     |     <div style="text-align: left"> `export PRECISION=fp32/fp16/bfloat16` </div> |
    |   <div style="text-align: left"> **MODEL_PATH** </div>   |    Local path to the downloaded model weights & tokenizer  |  <div style="text-align: left"> `export MODEL_PATH=/tmp/gemma_2b_en` </div> |
    |   <div style="text-align: left"> **KERAS_BACKEND** </div>   |     Determine the backend framework for Keras  | <div style="text-align: left"> `export KERAS_BACKEND=tensorflow/jax>` </div> |
    |   <div style="text-align: left"> **OUTPUT_DIR** </div>   |    Local path to save the output logs  | <div style="text-align: left"> `export OUTPUT_DIR=/tmp/keras_gemma_output` </div> |
    |   <div style="text-align: left"> **MAX_LENGTH (optional)** </div>   |     Max length of the generated sequence    | <div style="text-align: left"> `export MAX_LENGTH=64`</div> |
7. Run `run_model.sh`. This will run `N` instances of `generate.py`, where `N` is the number of sockets on the system (1 instance per socket).
    ```
    ./run_model.sh
    ```

## Output

Output of `run_model.sh` typically looks as below. Note that the value indicates the sum of throughput of all the instances:
```
Total throughput: 0.390845 inputs/sec
```

Output of any of the instances typically looks like:
```
Time taken for first generate (warmup): 10.724524021148682
Time taken for second generate: 10.216123819351196
Latency: 10.216123819351196 sec
Throughput: 0.1957689663286614 inputs/sec
```
followed by the `prompt` and its corresponding `output`.

Final results of the inference run can be found in `results.yaml` file.
```
results:
 - key: total throughput
   value: 0.390845
   unit: inputs/sec
```

## Additional Notes

- `keras_nlp` installs stock version of latest public `tensorflow` as a dependency. If you're running with a custom built or nightly version of TensorFlow, you will need to uninstall `tensorflow` after installing `keras-nlp` and then force reinstall your version of `tensorflow`.
- There are other ways to load the model using the Kaggle APIs like [`KaggleHub` or `Kaggle CLI` or `cURL`](https://www.kaggle.com/models/google/gemma/keras/) or by [configuring your Kaggle API key](https://ai.google.dev/gemma/docs/setup).
