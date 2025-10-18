# Financial Sentiment Analysis with LoRA

A comparative study of full fine-tuning vs LoRA (Low-Rank Adaptation) for financial sentiment classification using RoBERTa-base on the Financial PhraseBank dataset.

## Overview

This project explores parameter-efficient fine-tuning methods for financial sentiment analysis, comparing traditional full fine-tuning against LoRA with different rank configurations.

## Dataset

- **Source**: Financial PhraseBank (sentences_allagree subset)

- **Task**: 3-class sentiment classification (negative, neutral, positive)

- Split

  :

  - Train: 1,629 samples (72%)
  - Validation: 182 samples (8%)
  - Test: 453 samples (20%)

## Methods

### 1. Baseline (No Fine-tuning)

- Pretrained RoBERTa-base without any training
- Used for comparison baseline

### 2. Full Fine-tuning

- All 124.6M parameters trained
- 3 epochs, learning rate 2e-5
- Batch size 16

### 3. LoRA Fine-tuning

- Only 2.54% parameters trained (3.2M for r=16)
- Target modules: query, key, value, dense
- Ablation study with r ∈ {4, 8, 16, 32}
- 3 epochs, learning rate 2e-4
- Batch size 16

## Hardware

- GPU: Tesla V100-SXM2-32GB
- Framework: PyTorch with Transformers and PEFT

## Project Structure

```
financial_sentiment_lora/
├── README.md
├── baseline.ipynb           # Pretrained RoBERTa baseline
├── full_finetune.ipynb      # Full fine-tuning
├── lora_finetune.ipynb      # LoRA fine-tuning with ablation
└── analysis.ipynb           # Results visualization and analysis
```

## Usage

All experiments are implemented as Jupyter notebooks. Simply open any notebook and run all cells:

### 1. Run Baseline (Optional)

```bash
jupyter notebook baseline.ipynb
# Then: Cell > Run All
```

### 2. Run Full Fine-tuning

```bash
jupyter notebook full_finetune.ipynb
# Then: Cell > Run All
```

### 3. Run LoRA Fine-tuning

```bash
jupyter notebook lora_finetune.ipynb
# Then: Cell > Run All
```

### 4. View Analysis and Visualizations

```bash
jupyter notebook analysis.ipynb
# Then: Cell > Run All
```

**Note:** All results are printed directly to the notebook output. No files are saved to disk.

## Dependencies

- transformers
- datasets
- peft
- torch
- scikit-learn
- pandas
- matplotlib
- seaborn

## Future Work

- Test on other financial text datasets
- Explore different LoRA configurations (alpha, dropout)
- Compare with other PEFT methods (Adapter, Prefix-tuning)
- Fine-tune on domain-specific financial language models

## License

This project is for educational and research purposes.