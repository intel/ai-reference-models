<!--- 30. Datasets -->
### Dataset
Download Microsoft Research Paraphrase Corpus (MRPC) data in cloned repository and save it inside `data` folder.
You can also use the helper script [download_glue_data.py](https://gist.github.com/vlasenkoalexey/fef1601580f269eca73bf26a198595f3) to download the data:

   ```
   # Obtain a copy of download_glue_data.py to the current directory
   wget https://gist.githubusercontent.com/vlasenkoalexey/fef1601580f269eca73bf26a198595f3/raw/db67cdf22eb5bd7efe376205e8a95028942e263d/download_glue_data.py
   python3 download_glue_data.py --data_dir ./data/ --tasks MRPC
   ```
