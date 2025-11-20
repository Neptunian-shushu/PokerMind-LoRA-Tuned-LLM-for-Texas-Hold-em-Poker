# PokerMind: LoRA-Tuned LLM for Texas Hold'em Poker

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/Neptunian-shushu/PokerMind-LoRA-Tuned-LLM-for-Texas-Hold-em-Poker)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/Neptunian-shushu/PokerMind-LoRA-Tuned-LLM-for-Texas-Hold-em-Poker)](https://github.com/Neptunian-shushu/PokerMind-LoRA-Tuned-LLM-for-Texas-Hold-em-Poker/stargazers)
[![SFT Status](https://img.shields.io/badge/SFT-Complete-success)](https://github.com/Neptunian-shushu/PokerMind-LoRA-Tuned-LLM-for-Texas-Hold-em-Poker)
[![PPO Status](https://img.shields.io/badge/PPO-In%20Progress-yellow)](https://github.com/Neptunian-shushu/PokerMind-LoRA-Tuned-LLM-for-Texas-Hold-em-Poker)

**CS6220 Data and Visual Analytics Project - Georgia Institute of Technology**

This repository contains the CS6220 course project implementing LoRA fine-tuning of large language models for Texas Hold'em poker decision making, enhanced with PPO (Proximal Policy Optimization) self-play reinforcement learning.

## Project Overview

This project uses a two-phase approach to train AI poker agents:

1. **Phase 1 - Supervised Fine-Tuning (SFT)**: Fine-tune Meta-Llama-3-8B using LoRA on the PokerBench dataset (110k expert poker decisions)
   - ✅ **Status**: Complete
   - **Accuracy**: 35-45% action accuracy achieved

2. **Phase 2 - PPO Self-Play**: Further improve the model through reinforcement learning via self-play against itself
   - 🚧 **Status**: In Progress
   - **Target**: 55-65%+ accuracy

The project demonstrates how combining supervised learning with reinforcement learning can create competitive poker-playing AI agents using efficient fine-tuning techniques.

## Quick Start

### Prerequisites

Ensure you have Python 3.8+ and access to a CUDA-compatible GPU for training.

### Clone the Repository

```bash
git clone https://github.com/Neptunian-shushu/PokerMind-LoRA-Tuned-LLM-for-Texas-Hold-em-Poker.git
cd PokerMind-LoRA-Tuned-LLM-for-Texas-Hold-em-Poker
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: If using PACE ICE cluster, load the required modules first:
```bash
module load anaconda3
pip install --user -r requirements.txt
```

### Run SFT Training

The SFT training is designed to run on PACE ICE cluster using SLURM with H100 GPUs:

```bash
cd sft/
sbatch setup.sbatch  # Submits SLURM job on PACE ICE H100 GPU
```

Alternatively, run the training script directly:
```bash
cd sft/
python train_poker_model.py
```

**Training Details:**
- Dataset: 110,000 poker hands (80k postflop + 30k preflop) from [PokerBench](https://huggingface.co/datasets/RZ412/PokerBench)
- Model: Meta-Llama-3-8B with LoRA (r=16, α=32)
- Training: 3 epochs on H100 80GB GPU (~4.6 hours)
- Results: Model saved to `poker-lora-model/Meta-Llama-3-8B/`

### Run PPO Training (In Progress)

PPO training components are currently under development:

```bash
cd ppo/
python train_ppo.py  # Coming soon
```

### Run Demo Interface

The demo interface allows you to play against the trained AI:

```bash
cd backend/
python aiService.py  # Start backend service
```

For frontend interface, see [`frontend/README.md`](frontend/README.md) for setup instructions.

## Project Structure

```
PokerMind-LoRA-Tuned-LLM-for-Texas-Hold-em-Poker/
├── sft/                          # Supervised Fine-Tuning (✅ Complete)
│   ├── train_poker_model.py      # SFT training script
│   └── setup.sbatch               # SLURM job for H100 GPU
│
├── ppo/                          # PPO Reinforcement Learning (🚧 In Progress)
│   ├── config.py                  # PPO hyperparameters
│   ├── rewards.py                 # Reward calculation with GAE
│   ├── agents.py                  # Agent wrappers (TODO)
│   ├── ppo_trainer.py             # PPO algorithm (TODO)
│   └── train_ppo.py               # Main PPO training loop (TODO)
│
├── poker_game/                   # Texas Hold'em Game Engine (✅ Complete)
│   ├── game_logic.py              # Core poker game logic
│   ├── game_state.py              # Game state representation
│   ├── hand_evaluator.py          # Hand ranking system
│   ├── deck.py                    # Card and deck management
│   └── README.md                  # Game engine documentation
│
├── backend/                      # Backend API Service
│   ├── aiService.py               # AI inference service
│   └── SETUP.md                   # Backend setup guide
│
├── frontend/                     # Web Frontend Interface
│   └── README.md                  # Frontend setup instructions
│
├── logs/                         # Training Logs
│   ├── sft/                       # SFT training outputs
│   └── ppo/                       # PPO training outputs
│
├── poker-lora-model/             # Trained Model Weights
│   └── Meta-Llama-3-8B/          # SFT LoRA adapter weights
│
└── requirements.txt              # Python dependencies
```

## Results & Metrics

### Supervised Fine-Tuning (SFT) - Phase 1 ✅

**Configuration:**
- Dataset: 110,000 poker hands (80k postflop + 30k preflop)
- Model: Meta-Llama-3-8B with LoRA (r=16, α=32, dropout=0.1)
- Training: 3 epochs on H100 80GB GPU
- Duration: ~4.6 hours
- Batch size: 128 (16 per device × 8 gradient accumulation steps)

**Results:**
- **Final Loss**: 0.2022 (89% improvement from initial 1.9058)
- **Train-Val Gap**: 0.0006 (minimal overfitting ✅)
- **Preflop Accuracy**: 34.7%
- **Postflop Accuracy**: 44.7%
- **Overall Accuracy**: ~42% exact match

**Analysis**: This accuracy is expected and normal for 3 epochs of training on complex poker decision-making. The model has learned fundamental poker patterns and is ready for PPO enhancement.

### PPO Self-Play - Phase 2 🚧 (Planned)

**Method:**
- Policy optimization with frozen reference model
- KL divergence constraint (β = 0.1) to prevent policy collapse
- 10,000-50,000 self-play games

**Expected Results:**
- Target accuracy: **55-65%+** (+15-20% improvement over SFT)
- Better strategic play (aggression, bluffing, adaptation)
- Win rate vs frozen SFT opponent: 60%+

## Hardware & Resources

### Recommended Hardware

- **SFT Training**: NVIDIA H100 80GB or A100 40GB+
- **PPO Training**: NVIDIA H100 80GB (recommended)
- **Inference/Demo**: NVIDIA RTX 3090 24GB or any GPU with 8GB+ VRAM
- **Storage**: 20GB+ free space for models and datasets

### Minimum Requirements

- **CPU**: Multi-core processor
- **RAM**: 16GB+ recommended (8GB minimum)
- **GPU**: CUDA-compatible GPU with 8GB+ VRAM
- **Storage**: 20GB+ free space

## Dependencies & Installation

### Core Dependencies

The project uses the following key packages:

- **PyTorch, Transformers, PEFT**: ML frameworks and LoRA fine-tuning
- **BitsAndBytes**: 4-bit quantization for memory efficiency
- **Datasets, Accelerate**: Data handling and distributed training
- **NumPy, Pandas, Matplotlib**: Data processing and visualization
- **FastAPI, Uvicorn**: Backend API service

### Installation

Install all dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

For detailed installation instructions and troubleshooting, see `backend/SETUP.md`.

### PACE ICE Cluster Setup

If using Georgia Tech's PACE ICE cluster:

```bash
module load anaconda3
pip install --user -r requirements.txt
```

**Note**: PyTorch is pre-installed via PACE modules. Ensure CUDA modules are loaded for GPU support.

## What's Next / Roadmap

### Phase 1: Supervised Fine-Tuning
- [x] Dataset preparation and preprocessing
- [x] LoRA fine-tuning implementation
- [x] Training on H100 GPU cluster
- [x] Model evaluation and metrics
- [x] Save trained model weights

### Phase 2: PPO Self-Play (🚧 In Progress)
- [ ] Implement agent wrappers (`ppo/agents.py`)
- [ ] Implement PPO trainer (`ppo/ppo_trainer.py`)
- [ ] Implement training loop (`ppo/train_ppo.py`)
- [ ] Test with 1,000 episodes
- [ ] Run full training with 10,000-50,000 episodes
- [ ] Evaluate PPO model vs SFT baseline

### Phase 3: Demo & Evaluation
- [ ] Complete Gradio web interface
- [ ] Implement human vs AI gameplay
- [ ] Add game state visualization
- [ ] Test on held-out scenarios
- [ ] Compare SFT vs PPO performance
- [ ] Analyze learned strategies and behaviors

## Contributing

This is a course project for CS6220 at Georgia Institute of Technology. 

**For Team Members:**
- Coordinate through established team communication channels
- Follow the existing code structure and conventions
- Update documentation when making significant changes

**For External Contributors:**
- Open an issue to discuss proposed changes
- Submit pull requests with clear descriptions
- Follow the project's coding standards

## Citation

If you use this work in your research or projects, please cite:

```bibtex
@misc{pokermind2025,
  author = {bshu30},
  title = {PokerMind: LoRA-Tuned LLM for Texas Hold'em with PPO Self-Play},
  year = {2025},
  publisher = {Georgia Institute of Technology},
  journal = {CS6220 Course Project},
  url = {https://github.com/Neptunian-shushu/PokerMind-LoRA-Tuned-LLM-for-Texas-Hold-em-Poker}
}
```

## Contact

- **GitHub Issues**: [Open an issue](https://github.com/Neptunian-shushu/PokerMind-LoRA-Tuned-LLM-for-Texas-Hold-em-Poker/issues)
- **Email**: bshu30@gatech.edu
- **Repository**: [github.com/Neptunian-shushu/PokerMind-LoRA-Tuned-LLM-for-Texas-Hold-em-Poker](https://github.com/Neptunian-shushu/PokerMind-LoRA-Tuned-LLM-for-Texas-Hold-em-Poker)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details (if available).

## Links & Resources

- **Dataset**: [RZ412/PokerBench on Hugging Face](https://huggingface.co/datasets/RZ412/PokerBench)
- **Base Model**: [meta-llama/Meta-Llama-3-8B on Hugging Face](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
- **Training Cluster**: PACE ICE (Georgia Tech)
- **Game Engine Documentation**: [poker_game/README.md](poker_game/README.md)
- **Training Logs**: [logs/](logs/)

---

**Project Status**: Phase 1 (SFT) ✅ Complete | Phase 2 (PPO) 🚧 In Progress | Phase 3 (Demo) ⏳ Planned
