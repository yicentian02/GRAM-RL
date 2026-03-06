This repository contains the official implementation of the paper:

**Reinforcement Learning-Driven Generative Retrieval with Semantic-aligned Multi-Layer Identifiers**

---

# Overview

The training pipeline consists of two main stages:

1. **Supervised Fine-Tuning (SFT)**
   Train the model to generate semantic identifiers from queries.

2. **Reinforcement Learning (GRPO)**
   Optimize the generator using retrieval-based reward signals.

During training and evaluation, document embeddings are generated using **BGE (bge-en-v1.5)**.

---

# Environment Setup

We recommend using **Python ≥ 3.9**.

Main libraries used in this project:

* LLaMA-Factory (for SFT)
* TRL (for GRPO training)
* FlagEmbedding (for BGE embeddings)
* PyTorch
* Transformers

---

# Dataset

The **corpus**, **SFT data**, and **GRPO training data** can be downloaded from:

https://drive.google.com/drive/folders/1Hr972mBnmw3uJF7tUJR4T9yjhLP09eTH?usp=drive_link

The dataset contains:

* **Corpus** – document collection used for retrieval
* **SFT Data** – supervised training dataset
* **GRPO Data** – reinforcement learning training dataset

---

# Prompt Templates

Before training, you must **add prompts to the dataset**.

All prompt templates are provided in:

```
prompts/
```

These prompts are used to guide the model to generate semantic identifiers.

---

# Training

## Stage 1: Supervised Fine-Tuning (SFT)

The SFT stage is implemented using **LLaMA-Factory**.

This stage trains the model to generate identifiers (summary, keywords, pseudo-queries) conditioned on queries.

Example workflow:

1. Prepare the SFT dataset
2. Add prompts from `prompts/`
3. Run training with LLaMA-Factory

---

## Stage 2: GRPO Training

The reinforcement learning stage is implemented using **TRL** with **Group Relative Policy Optimization (GRPO)**.

Training script:

```
code/grpo/train.py
```

This stage optimizes the generator using reward signals derived from retrieval performance.

---

# Corpus Embedding Generation

Before running retrieval or GRPO training, you must generate embeddings for the corpus.

We use the **BGE embedding model**:

```
bge-en-v1.5
```

The embeddings are used during:

* candidate retrieval
* reward computation
* evaluation

---

# Evaluation

Evaluation scripts are provided in:

```
eval/eval.py
```

The evaluation pipeline measures retrieval performance using generated identifiers.

Reported metrics include:

* Recall@K
* MRR@K

---

# Repository Structure

```
├── prompts/
├── code/
│   └── grpo/
│       └── train.py
├── eval/
│   └── eval.py
└── README.md
```

---

# Citation

If you find this repository helpful, please cite our paper:

```
@inproceedings{GRAM,
author = {Xu, Bo and Tian, Yicen and Zhang, Xiaokun and Yu, Erchen and Li, Dailin and Zong, Linlin and Lin, Hongfei},
title = {Reinforcement Learning-Driven Generative Retrieval with Semantic-aligned Multi-Layer Identifiers},
year = {2025},
isbn = {9798400720406},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3746252.3761136},
doi = {10.1145/3746252.3761136},
pages = {3592–3601},
numpages = {10},
keywords = {generative retrieval, muti-layer identifier, reinforcement learning},
location = {Seoul, Republic of Korea},
series = {CIKM '25}
}
```

---

# Contact

If you have any questions, please open an issue in this repository.
