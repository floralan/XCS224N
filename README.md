# MiniGPT: Character-Level Language Model with Perceiver Architecture

A from-scratch implementation of a GPT-style transformer language model in PyTorch, featuring both vanilla self-attention and Perceiver-style cross-attention for efficient sequence modeling.

## Overview

This project implements a character-level GPT model based on [Andrej Karpathy's minGPT](https://github.com/karpathy/minGPT), extended with:

- **Causal Self-Attention**: Standard transformer attention with autoregressive masking
- **Perceiver Cross-Attention**: Efficient attention mechanism using learned latent bottleneck vectors, reducing sequence complexity from O(n²) to O(n·k)
- **Span Corruption Pretraining**: Self-supervised pretraining objective that masks random spans and trains the model to reconstruct them
- **Transfer Learning Pipeline**: Pretrain on Wikipedia text, finetune on downstream QA tasks

## Architecture

### Vanilla GPT
```
Input → Token Embedding + Position Embedding → [Transformer Block × N] → LayerNorm → Linear → Output
```

### Perceiver GPT
```
Input → Embedding → DownProject(C) → [Transformer Block × N-2] → UpProject → LayerNorm → Linear → Output
```

The Perceiver variant uses learned basis vectors `C` to compress the input sequence to a smaller bottleneck dimension, enabling more efficient attention computation.

**Model Configuration:**
- Layers: 4
- Attention Heads: 8
- Embedding Dimension: 256
- Block Size: 128
- Perceiver Bottleneck: 32

## Project Structure

```
├── src/
│   ├── submission/
│   │   ├── model.py          # GPT, Block, DownProjectBlock, UpProjectBlock
│   │   ├── attention.py      # CausalSelfAttention, CausalCrossAttention
│   │   ├── dataset.py        # CharCorruptionDataset, NameDataset
│   │   ├── trainer.py        # Training loop with learning rate scheduling
│   │   ├── helper.py         # Model initialization, pretrain/finetune functions
│   │   └── utils.py          # Utility functions
│   ├── run.py                # Main entry point
│   └── data/                 # Training data (not included)
└── requirements.txt
```

## Usage

### Installation
```bash
pip install -r requirements.txt
```

### Pretraining
```bash
# Vanilla GPT
python src/run.py pretrain vanilla wiki.txt --writing_params_path vanilla.pretrain.params

# Perceiver GPT
python src/run.py pretrain perceiver wiki.txt --writing_params_path perceiver.pretrain.params
```

### Finetuning
```bash
# With pretrained weights
python src/run.py finetune vanilla wiki.txt \
    --reading_params_path vanilla.pretrain.params \
    --writing_params_path vanilla.finetune.params \
    --finetune_corpus_path birth_places_train.tsv

# Without pretraining (from scratch)
python src/run.py finetune vanilla wiki.txt \
    --writing_params_path vanilla.model.params \
    --finetune_corpus_path birth_places_train.tsv
```

### Evaluation
```bash
python src/run.py evaluate vanilla wiki.txt \
    --reading_params_path vanilla.finetune.params \
    --eval_corpus_path birth_dev.tsv \
    --outputs_path vanilla.pretrain.dev.predictions
```

## Key Implementations

### Span Corruption (`dataset.py`)
Self-supervised pretraining that randomly masks spans in the input:
```
Input:  "Khatchig Mouradian is a journalist"
Output: "Khatchig Mouradian is a jour⁇and tran⁇nalist, writer⁇"
```

### Cross-Attention (`attention.py`)
Efficient attention between two sequences with causal masking, enabling the Perceiver's bottleneck compression.

### Perceiver Blocks (`model.py`)
- **DownProjectBlock**: Projects input sequence to bottleneck using learned basis vectors
- **UpProjectBlock**: Projects bottleneck back to original sequence length

## Technologies

- **PyTorch** - Deep learning framework
- **Transformer Architecture** - Self-attention, multi-head attention, positional embeddings
- **Perceiver** - Efficient cross-attention with learned latents

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al.
- [minGPT](https://github.com/karpathy/minGPT) - Andrej Karpathy
- [Perceiver](https://arxiv.org/abs/2103.03206) - Jaegle et al.
