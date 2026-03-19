# Single-Cell Electrophysiology Analysis & Clustering


[![Paper DOI](https://img.shields.io/badge/Paper-DOI-blue)](https://doi.org/10.1371/journal.pcbi.1013821)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains the analysis code, Jupyter notebooks, and extracted feature files accompanying the following publications:  

Joshi N, van Der Burg S, Celikel T, Zeldenrust F (2025) Neuronal identity is not static: An input-driven perspective. PLoS Comput Biol 21(12): e1013821. https://doi.org/10.1371/journal.pcbi.1013821 

Neuromodulatory Control of Cortical Function: Cell-Type Specific Regulation of Neuronal Information Transfer
Nishant Joshi, Xuan Yan, Niccolo Calcini, Payam Safavi, Asli Ak, Koen Kole, Sven van der Burg, Tansu Celikel, Fleur Zeldenrust
bioRxiv 2026.03.13.711516; doi: https://doi.org/10.64898/2026.03.13.711516

---

## Overview
This project provides the computational workflows for analyzing **single-cell electrophysiological recordings** and reproduces the results presented in the associated paper.  

Key methods and analyses:  
- **Feature extraction** from electrophysiological recordings  
- **Model fitting** using GLIF models (see https://github.com/Nishant-codex/GIFFittingToolbox/tree/master)
- **Clustering** of single-cell responses across experimental conditions (ACSF vs. drug)  
- **Dimensionality reduction**: PCA, UMAP, and probabilistic CCA (pCCA)  
- **Manifold alignment** for comparing cell populations
- **MCFA** for comparing extracted features for heterogeneity  
- Exploratory analyses of **cluster stability**, **spike train metrics**, and **impedance profiles**  

---

## Repository Structure
```
notebooks_acsf/        # Analysis under ACSF conditions
notebooks_drug/        # Analysis under drug conditions
residual_code/         # Exploratory / legacy notebooks and scripts
│   └── Infomation_transfer/   # Cluster analysis, PCA, UMAP, stability checks
LICENSE
README.md              # Project description and usage
```

---

## Installation
Clone the repository and install dependencies:  

```bash
git clone https://github.com/your-username/single_cell_analysis-clustering.git
cd single_cell_analysis-clustering
pip install -r requirements.txt
```

Alternatively, create a conda environment:  

```bash
conda env create -f environment.yml
conda activate singlecell
```

---

## Usage
To reproduce the analyses, open the Jupyter notebooks:  

```bash
jupyter notebook notebooks_acsf/Clustering_attribute_sets.ipynb
```

- `notebooks_acsf/` → Main analysis for ACSF  
- `notebooks_drug/` → Drug condition analysis  
- `residual_code/` → Supplemental and exploratory analyses  

Figures and results generated from these notebooks correspond directly to the analyses presented in the paper.  

---

## Workflow
Below is a simplified workflow of the analysis pipeline:  

```mermaid
flowchart TD
    A[Raw Electrophysiology Data] --> B[Feature Extraction]
    B --> D[Clustering & Manifold Alignment]
	D --> E[Multi-set correlation and Factor Analysis]
    E --> F[Comparative Analysis: ACSF vs Drug]
    F --> G[Figures & Results]
```

---

## Citation
If you use this code or workflows in your research, please cite:  

```

@article{joshi2025neuronal,
  title={Neuronal identity is not static: An input-driven perspective},
  author={Joshi, Nishant and van Der Burg, Sven and Celikel, Tansu and Zeldenrust, Fleur},
  journal={PLOS Computational Biology},
  volume={21},
  number={12},
  pages={e1013821},
  year={2025},
  publisher={Public Library of Science San Francisco, CA USA}
}
```

---

## License
This project is licensed under the [MIT License](LICENSE).  
