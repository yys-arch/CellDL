# CellDL: Defining Cell Identity by Learning Transcriptome Distributions

[![PyPI version](https://badge.fury.io/py/CellDL.svg)](https://badge.fury.io/py/CellDL)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**CellDL** is a deep probabilistic representation learning framework designed to redefine how cell identity is modeled in single-cell RNA-seq (scRNA-seq) data.

## 📖 Introduction & Motivation

Cell identity defines what a cell is, how it functions, and what it can become. Currently, most computational approaches adopt a deterministic paradigm, compressing the cellular transcriptional state into a single, fixed vector. This approach treats a dynamic, stochastic entity as a static point, discarding the variability and uncertainty essential to biological nature.

**CellDL moves from point estimates to probabilistic representations.**

It represents each cell through a set of gene-wise probability distributions. By leveraging a decoupled deep learning architecture, CellDL captures the full distribution of transcriptional states, preserving biological heterogeneity and variability that traditional methods miss.

## 🚀 Key Features

*   **Probabilistic Representation**: Models gene expression using parametric distributions (e.g., IZIP, ZINB) rather than fixed values.
*   **Decoupled Architecture**: Uses a shared encoder for global cell state and decoupled heads for inferring gene-specific distribution parameters ($\lambda$, $\phi$, etc.).
*   **Biologically-Informed Denoising**: Reconstructs expression profiles based on the expected values ($\mathbb{E}$) of learned distributions, effectively removing technical noise while keeping biological signals.
*   **Generative Data Augmentation**: Generates realistic synthetic cells via controlled perturbation of learned parameters, facilitating the analysis of rare cell populations.

## 🛠 Model Architecture

<div align="center">
  <img src="docs/fig1.png" alt="CellDL Architecture" width="800"/>
  <p><em>Figure 1: Schematic of the CellDL model architecture. The model maps cells to a latent embedding and decodes them into gene-specific distributional parameters.</em></p>
</div>

CellDL employs a **decoupled autoencoder architecture**:
1.  **Encoder**: Maps the raw count matrix to a non-linear latent embedding.
2.  **Decoupled Decoders**: Independently infer the parameters of the underlying distribution (e.g., Mean expression rate $\lambda$ and Dropout probability $\phi$).
3.  **Objective**: Minimizes the difference between the expected value of the predicted distribution and the observed data using a self-supervised expectation-based loss.

## 📦 Installation

### Install from PyPI
```bash
pip install celldl==0.1.2
```

### Install from Source
```bash
git clone https://github.com/yys-arch/CellDL.git
cd CellDL
pip install .
```

**Requirements:** Python >= 3.10, TensorFlow, Scanpy, AnnData, etc.

## 💻 Usage Tutorial

### 1. Data Preprocessing
CellDL provides a robust preprocessing pipeline including HVG selection and normalization.

```python
import scanpy as sc
from CellDL import data_preprocessing

# Load data
adata = sc.read_h5ad("your_data.h5ad")

# Preprocess: Filter, Log-normalize, and Select HVGs
adata = data_preprocessing(
    adata, 
    assay="10x 3' v3",   # Optional filtering
    gene_mean_min=0.0125,
    gene_mean_max=3,
    gene_disp_min=0.5
)
```

### 2. Model Training
Initialize and train the model using one of the supported distribution modes. The paper highlights the **IZIP (Independent Zero-Inflated Poisson)** mode.

```python
from CellDL import build_model, train_model, save_trained_model

# Build model with IZIP distribution (Recommended)
model = build_model(adata, mode='IZIP_mode', bottle_dim=512)

# Train
history = train_model(model, adata, epochs=1000, batch_size=32)

# Save
save_trained_model(model, 'models/celldl_model.keras')
```

### 3. Denoising (Signal Reconstruction)
Reconstruct gene expression using the expected value of the inferred distribution.

```python
from CellDL import load_trained_model, denoise_data

model = load_trained_model('models/celldl_model.keras')
adata_denoised = denoise_data(model, adata)

# Result is stored in .obsm
print(adata_denoised.obsm['rna_denoised'])
```

### 4. Synthetic Data Generation (Sample Expansion)
Generate synthetic cells to augment rare populations by perturbing the learned parameters.

```python
from CellDL import generate_sc_synthetic_data

# Generate 5 synthetic cells for every original cell
adata_synthetic = generate_sc_synthetic_data(model, adata, num_samples=5, deviation_scale=0.1)
```

## 📊 Supported Distributions

While the manuscript focuses on IZIP, the package supports multiple distribution families to fit different data characteristics:

*   `IZIP_mode`: Independent Zero-Inflated Poisson (**Default**)
*   `ZINB_mode`: Zero-Inflated Negative Binomial
*   `NB_mode`: Negative Binomial
*   `Mix_P_NB_mode`: Mixture of Poisson and NB
*   (See documentation for full list of mixture models)

## 📂 Data Availability

The datasets used in our manuscript to benchmark and validate CellDL are publicly available through the [CZ CELLxGENE Discover](https://cellxgene.cziscience.com/) platform.

| Dataset / Tissue | File Name / Description | Source Link |
| :--- | :--- | :--- |
| **Heart** | Tabula Sapiens - Heart | [Collection Link](https://cellxgene.cziscience.com/collections/e5f58829-1a66-40b5-a624-9046778e74f5) |
| **Bladder** | Tabula Sapiens - Bladder | [Collection Link](https://cellxgene.cziscience.com/collections/e5f58829-1a66-40b5-a624-9046778e74f5) |
| **Breast** | scRNA-seq data - all cells | [Collection Link](https://cellxgene.cziscience.com/collections/4195ab4c-20bd-4cd3-8b3d-65601277e731) |
| **Bone Marrow** | Fetal Bone Marrow (10x) | [Blood and immune development...](https://cellxgene.cziscience.com/) |
| **Large Intestine**| Tabula Sapiens - Large_Intestine | [Collection Link](https://cellxgene.cziscience.com/collections/e5f58829-1a66-40b5-a624-9046778e74f5) |
| **Lung** | Tabula Sapiens - Lung | [Collection Link](https://cellxgene.cziscience.com/collections/e5f58829-1a66-40b5-a624-9046778e74f5) |
| **Skin** | Skin | [Collection Link](https://cellxgene.cziscience.com/collections/43d4bb39-21af-4d05-b973-4c1fed7b916c) |
| **Spleen** | Tabula Sapiens - Spleen | [Collection Link](https://cellxgene.cziscience.com/collections/e5f58829-1a66-40b5-a624-9046778e74f5) |
| **iPSC-Derived EBs**<br>(Wellington et al. 2024) | Developmental Regulation of Endothelium | [Collection Link](https://cellxgene.cziscience.com/collections/4a2c25af-558a-45fc-bc9a-54ec44a1d63f) |

## 📧 Contact

Email: yyusong526@gmail.com