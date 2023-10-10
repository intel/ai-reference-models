<!--- 0. Title -->
# MEMREC: Memory Efficient Recommendation System using Alternative Representation

<!-- 10. Description -->
## Description

Deep learning-based recommendation systems (e.g., DLRMs) are widely used AI models to provide high-quality personalized recommendations. Training data used for modern recommendation systems commonly includes categorical features taking on tens-of-millions of possible distinct values. These categorical tokens are typically assigned learned vector representations, that are stored in large embedding tables, on the order of 100s of GB. Storing and accessing these tables represent a substantial burden in commercial deployments. Our work proposes MEM-REC, a novel alternative representation approach for embedding tables. MEM-REC leverages bloom filters and hashing methods to encode categorical features using two cache-friendly embedding tables. The first table (token embedding) contains raw embeddings (i.e. learned vector representation), and the second table (weight embedding), which is much smaller, contains weights to scale these raw embeddings to provide better discriminative capability to each data point. We provide a detailed architecture, design and analysis of MEM-REC addressing trade-offs in accuracy and computation requirements, in comparison with state-of-the-art techniques. We show that MEM-REC can not only maintain the recommendation quality and significantly reduce the memory footprint for commercial scale recommendation models but can also improve the embedding latency. In particular, based on our results, MEM-REC compresses the MLPerf CriteoTB benchmark DLRM model size by 2900x and performs up to 3.4x faster embeddings while achieving the same AUC as that of the full uncompressed model.

## Details 
Please refer to [MEMREC reseasrch paper (to appear in ACML'23)](https://arxiv.org/pdf/2305.07205.pdf) available on arxiv.

## Important MEMREC Parameters
  - ```D```: Size of the sparse binary vector - Tier-1 embedding table
  - ```K```: Number of Hash Functions - Tier-1 embedding table
  - ```dw```: Size of the sparse binary vector - Tier-2 embedding table
  - ```kw```: Number of Hash Functions - Tier-2 embedding table
  - ```arch-mlp-bot-sparse```: MLP that learns on the top of HD-Encoded data
  - ```arch-mlp-bot```: DLRM Dense processing MLP
  - ```arch-mlp-top```: DLRM predictor MLP


<!--- 80. License -->
## License

[LICENSE](../product/LICENSE)

[THIRD PARTY LICENSES](../product/THIRD-PARTY-LICENSES)

