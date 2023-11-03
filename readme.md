# mmAAVI (Multi-omics Mosaic Auto-scaling Attention Variational Inference)

A deep generative model that addresses the mosaic integration challenges.

![Figure 1](asset/Figure1-V4.png)

**Figure 1.** Schematic of mosaic integration and mmAAVI. **a,** Graph illustration of mosaic integration data with 3 batches and 3 modalities. Batch effects may arise from differences in individuals, experimental conditions, or tissue sources. Potential modalities include chromatin accessibility (DNA-level), gene expression (RNA-level), and epitope (protein-level), among others. Due to the combined impact of batch effects and modalities missingness, biological variations (such as cell types) are obscured by systematic errors, making direct integration analysis challenging. **b-c,** The workflow for unsupervised (**b**) and semi-supervised (**c**) analysis using mmAAVI. **d,** Schematic of mmAAVI model. Multi-modal data for each cell $n$ are transformed into embeddings by modality-specific encoders and then fused into a global feature $z_n$, a low-dimensional representation of the cell state following mixture distribution parameterized by discrete $c_n$ and continue $u_n$. A discriminator is used to harmonize the distribution of $z_n$ across different batches $s_n$. Meanwhile, a guidance graph $\mathcal{G}$ with prior knowledge is transformed into feature embeddings $V$ by a graph encoder. Next, modality-specific hybrid decoders map samples from the posterior distribution of $z_n$ and $V$, along with the batch, $s_n$, to parameters of the distribution for each feature of existed modalities. The posterior mean of $z_n$ can be used as input to clustering and visualization algorithms.

[toc]

## Directory structure

```
.
├── asset                   # Files to shown in readme
├── data                    # Data files with the scripts to download them
├── enviroments             # Scripts and packages lists to create reproducible python and R enviroments
├── experiments             # Codes for experiments and case studies
├── src/mmAAVI              # Main Python codes to create mmAAVI model
├── tests                   # Some test codes
└── readme.md
```

## Usage

## Development

## Reproduce results
