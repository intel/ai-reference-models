<!-- 70. Model args -->
Note that args specific to this model are specified after ` -- ` at
the end of the command (like the `profile=True` arg in the Profile
command above. Below is a list of all of the model specific args and
their default values:

| Model arg | Default value |
|-----------|---------------|
| doc_stride | `128` |
| max_seq_length | `384` |
| profile | `False` |
| config_file | `bert_config.json` |
| vocab_file | `vocab.txt` |
| predict_file | `dev-v1.1.json` |
| init_checkpoint | `model.ckpt-3649` |
