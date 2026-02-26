# BFCS: A Large-Scale Execution-Based Benchmark for Function Calling in Science
This repository contains the official implementation of the paper: **"BFCS: A Large-Scale Execution-Based Benchmark for Function Calling in Science"**.

## 🌟 Overview

**BFCS** is the first execution-based benchmark specifically designed to evaluate the function-calling capabilities of Large Language Models (LLMs) in scientific domains.
Unlike static benchmarks, BFCS adopts an **execution-first philosophy**:

- **Real-World Scale:** Includes **1,648 function-query-answer pairs** across chemistry, biology, pharmacy, medicine, and materials science.
- **Standardized Environment:** Integrated with **48 real scientific Python libraries** (e.g., RDKit, Biopython) and **2,100 executable tools**.
- **Rigorous Evaluation:** Uses **Apptainer** for container-native isolation to ensure reproducibility and verify functional correctness (ESR) and semantic accuracy (AMR).

## 📊 Main Results

| Model | Simple ESR | Simple AMR | Simple Gap↓ | Multiple ESR | Multiple AMR | Multiple Gap↓ | Parallel ESR | Parallel AMR | Parallel Gap↓ | **Overall ESR** | **Overall AMR** | **Overall Gap↓** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ***Proprietary Models*** | | | | | | | | | | | | |
| **Claude-Opus-4.5** | **98.94** | **69.74** | 29.21 | **99.39** | **65.97** | 33.42 | 93.85 | **75.89** | **17.96** | **97.39** | **70.53** | 26.86 |
| **Claude-Sonnet-4.5** | 97.71 | 60.11 | 37.60 | 95.09 | 63.82 | 31.27 | **95.47** | 68.47 | 27.00 | 96.09 | 64.13 | 31.96 |
| **Gemini-3-Pro** | 97.15 | 69.69 | **27.47** | 93.81 | 63.03 | 30.78 | 92.28 | 73.30 | 18.99 | 94.42 | 68.67 | **25.74** |
| **Gemini-3-Flash** | 94.47 | 64.50 | 29.96 | 92.18 | 61.09 | 31.09 | 85.71 | 65.68 | 20.03 | 90.79 | 63.76 | 27.03 |
| **GPT-5.2** | 92.80 | 65.23 | 27.56 | 93.13 | 64.44 | **28.69** | 92.41 | 61.12 | 31.28 | 92.78 | 63.60 | 29.18 |
| **Doubao-Seed-1.8** | 95.61 | 50.19 | 45.42 | 98.18 | 36.00 | 62.18 | 93.21 | 43.73 | 49.48 | 95.67 | 43.31 | 52.36 |
| ***Open-Weight Models*** | | | | | | | | | | | | |
| **DeepSeek-V3.2** | 89.12 | 54.77 | **34.35** | 91.27 | 54.91 | 36.36 | **98.26** | 4.53 | 93.73 | 92.88 | 38.07 | 54.81 |
| **GLM-4.7** | 91.27 | 48.28 | 42.99 | 96.26 | **61.07** | **35.19** | 91.02 | **58.23** | **32.79** | 92.85 | 55.86 | **36.99** |
| **Kimi-k2.5** | 91.98 | 44.85 | 47.14 | 89.64 | 29.09 | 60.55 | 84.84 | 40.42 | 44.43 | 88.82 | 38.12 | 50.70 |
| **Mistral-Large-3** | **100.00** | 48.28 | 51.72 | **100.00** | 49.64 | 50.36 | 91.29 | 43.90 | 47.39 | 97.10 | 47.27 | 49.82 |
| **Qwen3-235B** | 97.62 | **59.21** | 38.42 | 99.69 | 60.58 | 39.12 | 94.58 | 57.69 | 36.89 | **97.30** | **59.16** | 38.14 |
| **Qwen3-30B** | 93.13 | 42.07 | 51.06 | 97.49 | 38.63 | 58.86 | 90.54 | 34.24 | 56.30 | 93.72 | 38.31 | 55.41 |

*Note: ESR (Execution Success Rate) measures if the code runs; AMR (Answer Match Rate) measures if the scientific logic is correct. Gap = ESR − AMR, where a positive gap indicates potential silent failures. All values are in percentages.*

## 🚀 Getting Started

### 1. Prerequisites
We use **Apptainer** (formerly Singularity) to manage complex scientific dependencies.

```bash
# Install Apptainer (refer to official docs for details)
sudo apt-get update && sudo apt-get install -y apptainer

```

### 2. Download Data & Containers
The benchmark is stratified into three scenarios:

- **Simple:** Atomic instruction synthesis.
- **Multiple:** Tool selection among distractors.
- **Parallel:** Compositional batch processing.

```bash
git clone https://github.com/ChampionZhong/BFCS.git
cd BFCS
# Build scientific environment containers
bash containers/build_apptainers_*.sh

```

### 3. Run Evaluation

```bash
python evalution/run_eval.py --model_name your_model_name --scenario simple

```

## 📂 Dataset Taxonomy
BFCS covers a wide range of scientific disciplines:

- **Chemistry:** (e.g., `pyscf`, `rdkit`) 
- **Biology:** (e.g., `biopython`, `scanpy`) 
- **Medicine/Pharmacy:** (e.g., `monai`, `tdc`) 
- **Material Science:** (e.g., `gpaw`, `dscribe`) 

## ✍️ Citation
If you find this work helpful, please cite our KDD 2026 paper:
<!-- booktitle={Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining}, -->
```code snippet
@misc{zhong2026bfcs,
  title={BFCS: A Large-Scale Execution-Based Benchmark for Function Calling in Science},
  author={Zhong, Zhanping and Su, Xuerui and Zhang, Wei and Pei, Qizhi and Wang, Zun and He, Conghui and Wu, Lijun},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{[https://github.com/ChampionZhong/BFCS](https://github.com/ChampionZhong/BFCS)}},
  year={2026},
}

```
