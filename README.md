# PokerMind: LoRA-Tuned LLM for Texas Hold'em Poker

This repository contains the CS6220 (Data and Visual Analytics) project implementing LoRA fine-tuning of large language models for Texas Hold'em poker decision making.

## Project Overview

This project fine-tunes language models (Gemma-2B and Llama-3-8B) using LoRA (Low-Rank Adaptation) on the RZ412/PokerBench dataset to create an AI system capable of making strategic poker decisions in both preflop and postflop scenarios.

## Quick Start

### 1. Environment Setup
```bash
# Create conda environment
conda create -n cs6220 python=3.9 -y
conda activate cs6220

# Install packages
pip install -r requirements.txt
```

### 2. Run the Project
```bash
jupyter notebook poker_finetuning.ipynb
```

## Files Structure

- `poker_finetuning.ipynb`: Main notebook with complete LoRA fine-tuning implementation
- `requirements.txt`: Python package dependencies
- `SETUP.md`: Detailed setup instructions for teammates
- `README.md`: This file

## Package Requirements

Core dependencies include:
- **PyTorch, Transformers, PEFT**: ML frameworks and fine-tuning
- **NumPy, Pandas, Matplotlib**: Data analysis and visualization
- **Jupyter, Datasets, BitsAndBytes**: Development and optimization

See `requirements.txt` for complete list and `SETUP.md` for detailed installation guide.

## Hardware Requirements

- **Minimum**: 8GB RAM, multi-core CPU
- **Recommended**: 16GB+ RAM, NVIDIA GPU with 8GB+ VRAM
- **Storage**: 20GB+ free space

## Team Setup

For teammates to get started:
1. Clone this repository
2. Follow instructions in `SETUP.md`
3. Install packages from `requirements.txt`
4. Open and run `poker_finetuning.ipynb`

## Models Supported

- **Gemma-2B**: Lightweight model for testing and development
- **Llama-3-8B**: Full-scale model for production-quality results

## Features

- LoRA fine-tuning for memory-efficient training
- 4-bit quantization support for large models
- Preflop vs Postflop performance analysis
- Comprehensive evaluation metrics
- Model saving and export functionality

## Contributing

This is a course project for CS6220 at Georgia Tech. Team members should coordinate through the established communication channels.