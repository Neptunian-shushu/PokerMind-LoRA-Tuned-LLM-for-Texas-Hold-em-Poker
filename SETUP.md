# Package Installation Guide

## Essential Packages for PokerMind Project

This document lists all necessary Python packages for the PokerMind: LoRA-Tuned LLM for Texas Hold'em Poker project.

## Quick Installation

### Option 1: Install from requirements.txt (Recommended)
```bash
pip install -r requirements.txt
```

### Option 2: Conda Environment Setup (Recommended for CS6220)
```bash
# Create conda environment
conda create -n cs6220 python=3.9 -y
conda activate cs6220

# Install core packages
conda install numpy pandas matplotlib seaborn scikit-learn jupyter notebook ipykernel -y

# Install ML packages via pip (some not available in conda)
pip install transformers torch peft datasets bitsandbytes accelerate tqdm
```

## Package Categories and Purpose

### 1. Core Machine Learning Frameworks
- **torch** (>=2.0.0): PyTorch deep learning framework
- **transformers** (>=4.30.0): Hugging Face transformers library for LLMs
- **datasets** (>=2.12.0): Hugging Face datasets library
- **accelerate** (>=0.20.0): Distributed training and optimization

### 2. Fine-tuning and Quantization
- **peft** (>=0.4.0): Parameter Efficient Fine-Tuning (LoRA implementation)
- **bitsandbytes** (>=0.39.0): 4-bit/8-bit quantization for memory efficiency

### 3. Data Science and Visualization
- **numpy** (>=1.24.0): Numerical computing
- **pandas** (>=2.0.0): Data manipulation and analysis
- **matplotlib** (>=3.7.0): Basic plotting
- **seaborn** (>=0.12.0): Statistical visualization
- **scikit-learn** (>=1.3.0): Machine learning utilities

### 4. Jupyter Environment
- **jupyter** (>=1.0.0): Jupyter ecosystem
- **notebook** (>=6.5.0): Jupyter notebooks
- **ipykernel** (>=6.23.0): Jupyter kernel for Python
- **ipywidgets** (>=8.0.0): Interactive widgets

### 5. Utilities
- **tqdm** (>=4.65.0): Progress bars
- **packaging** (>=23.0): Package version handling

## Hardware Requirements

### Minimum Requirements
- **CPU**: Multi-core processor
- **RAM**: 8GB+ (16GB recommended)
- **Storage**: 10GB free space
- **Python**: 3.8+ (3.9 recommended)

### Recommended for Training
- **GPU**: NVIDIA GPU with 8GB+ VRAM (for Llama-3-8B training)
- **RAM**: 16GB+ system memory
- **Storage**: 20GB+ free space (for model checkpoints)

## Platform-Specific Notes

### PACE Cluster (Georgia Tech)
```bash
# Load required modules
module load anaconda3
conda create -n cs6220 python=3.9 -y
conda activate cs6220
```

### Local Development
- Ensure CUDA is installed for GPU support
- Consider using `conda` for better dependency management
- WSL2 recommended for Windows users

## Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce batch size or use quantization
2. **Package conflicts**: Use virtual environments
3. **Slow downloads**: Use `--trusted-host` for pip behind firewalls

### Version Compatibility
- PyTorch version should match your CUDA version
- Transformers and PEFT versions should be compatible
- Use `pip check` to verify package compatibility

## Optional Packages

For extended functionality, consider installing:
- **scipy**: Scientific computing functions
- **plotly**: Interactive visualizations  
- **wandb**: Experiment tracking and logging
- **tensorboard**: TensorBoard integration

## Contact

For setup issues or questions, contact the project team or refer to the main README.md file.