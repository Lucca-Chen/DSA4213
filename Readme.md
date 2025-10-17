⚠️ **NOTE:** IF YOU WANT TO REPRODUCE THE CODE, PLEASE MAKE SURE TO DELETE THE `CONFIG` AND `LOG` FOLDERS FIRST.  AFTER RUNNING `00_SETUP_ENV.IPYNB`, THE CORRESPONDING FOLDERS WILL BE AUTOMATICALLY GENERATED.Legal Question Answering with Parameter-Efficient Fine-tuning

**NUS DSA4213 Assignment 3 - Fine-tuning Pretrained Transformers**

**Student ID:** A0327241M  
**Name:** Chen Zhaoyun  
**Semester:** 2025/2026 Semester 1

---

## 📊 Project Overview

This project compares three fine-tuning strategies for legal question answering:

- **Full Fine-tuning**: All 124M parameters trainable
- **LoRA (Low-Rank Adaptation)**: Parameter-efficient with ranks r=4, 8, 16
- **Prompt Tuning**: Extremely lightweight with learnable soft prompts

**Dataset:** LegalQAEval (2,534 samples, 51.6% unanswerable questions)  
**Model:** RoBERTa-base (124M parameters)  
**Task:** Extractive QA with "no answer" detection

---

## 📈 Key Results

| Method | EM (%) | F1 (%) | Trainable Params | Model Size |
|--------|--------|--------|------------------|------------|
| **Full FT** | 75.21 | 76.53 | 124M (100%) | 1,419.80 MB |
| **LoRA r=16** | 67.72 | 69.28 | 589K (0.48%) | 6.80 MB |
| **Prompt Tuning** | 66.94 | 68.81 | 50K (0.04%) | ~1.50 MB |

**Key Findings:**
- LoRA achieves 90% of full fine-tuning performance with only 0.48% trainable parameters
- 99.5% reduction in model size with LoRA
- Prompt Tuning enables efficient multi-task deployment

---

## 📁 Project Structure

```
DSA4213/
├── data/                           # Dataset files
├── configs/                        # Configuration files
├── logs/                          # Training logs
├── outputs/                       # Model checkpoints and predictions
├── reports/                       # Generated analysis reports
├── utils.py                       # Helper functions
├── 00_setup_env.ipynb            # Environment setup
├── 01_data_prep.ipynb            # Data preprocessing
├── 02_train_roberta.ipynb        # Full FT & LoRA training
├── 03_RoBERTa_Prompt_Tu....ipynb # Prompt tuning training
├── data_validation_report....    # Data analysis report
└── README.md                      # This file
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Lucca-Chen/DSA4213
cd DSA4213

# Install dependencies
pip install -r requirements.txt
```

**Required packages:**
- torch>=2.0.0
- transformers>=4.36.0
- datasets
- peft
- accelerate
- numpy
- pandas
- scikit-learn
- jupyter

---

## 🔬 Running Experiments

**All experiments are organized as Jupyter notebooks. Simply run them in order:**

### Step 1: Setup Environment
```bash
jupyter notebook 00_setup_env.ipynb
```
Run all cells to verify installation and setup paths.

### Step 2: Data Preprocessing
```bash
jupyter notebook 01_data_prep.ipynb
```
Run all cells to load and preprocess the LegalQAEval dataset.

### Step 3: Train Full Fine-tuning & LoRA
```bash
jupyter notebook 02_train_roberta.ipynb
```
Run all cells to:
- Train full fine-tuning baseline
- Train LoRA with ranks 4, 8, 16
- Generate comparison metrics

### Step 4: Train Prompt Tuning
```bash
jupyter notebook 03_RoBERTa_Prompt_Tu....ipynb
```
Run all cells to train and evaluate prompt tuning.

**That's it!** All results will be saved in the `outputs/` and `reports/` directories.

---