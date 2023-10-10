# MEMREC Encoder: 

MEMREC Encoder encodes categorical tokens into a large binary vector using wyhash.

## Installation

To compile and install the module use:

`python setup.py install --user`

## Instructions

Usage:

`import memrec_encoder`

`memrec_encoder.wyh_hash_array_sparse(input_list, D, n_bit_hash, featureCnt, K, kw, hash_tech_id)`

The arguments are as follows:
- `input_list` - A numpy array containing the data to be hashed. 
- `n_bit_hash` - For hashing function to generate n bit hases.
- `D` - The encoding dimension to be used. Currently same for both embedding tables.
- `featureCnt` - Feature count, i.e. number of columns in the data (26 for Criteo TB).
- `K and kw` - Number of hashes for each embedding table.
- `hash_tech_id` - 2 for wy hash.

