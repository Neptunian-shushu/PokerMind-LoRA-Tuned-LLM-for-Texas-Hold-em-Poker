# PokerMind: LoRA-Tuned LLM for Texas Hold'em Poker

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)

CS6220 Data and Visual Analytics Project - Georgia Institute of Technology

A complete implementation of LoRA fine-tuning for large language models applied to Texas Hold'em poker decision making, enhanced with PPO (Proximal Policy Optimization) reinforcement learning and an interactive web-based demo.

## Project Overview

This project implements a three-phase approach to training AI poker agents:

1. **Phase 1 - Supervised Fine-Tuning (SFT)**: ✅ **COMPLETE** - Fine-tune Llama-3-8B using LoRA on 110k expert poker hands from the PokerBench dataset
2. **Phase 2 - PPO Self-Play**: ✅ **COMPLETE** - Enhance the model through reinforcement learning via self-play
3. **Phase 3 - Interactive Frontend**: ✅ **COMPLETE** - Web-based demo for playing against the trained AI agent

The project achieves **~42% exact-match action accuracy** on poker decisions after supervised training, with further improvements through PPO reinforcement learning.

## Quick Start

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended: 8GB+ VRAM for inference, 40GB+ for training)
- Node.js v22+ (for frontend)

### Installation

```bash
# Clone the repository
git clone https://github.com/Neptunian-shushu/PokerMind-LoRA-Tuned-LLM-for-Texas-Hold-em-Poker.git
cd PokerMind-LoRA-Tuned-LLM-for-Texas-Hold-em-Poker

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training the Model

#### Supervised Fine-Tuning (SFT)

Run SFT training on a SLURM cluster with GPU:

```bash
cd sft/
sbatch setup.sbatch  # Submits job to PACE ICE cluster with H100 GPU
```

Or run locally:

```bash
cd sft/
python train_poker_model.py
```

**Results**: The SFT phase achieves ~42% overall accuracy and saves the model to `poker-lora-model/`.

#### PPO Reinforcement Learning

Run PPO self-play training:

```bash
cd ppo/
python train_ppo.py
```

PPO training logs and checkpoints are saved to `logs/ppo/`.

### Running the Demo

The interactive frontend allows you to play poker against the trained AI agent. See [frontend/README.md](frontend/README.md) for detailed setup instructions.

```bash
cd frontend/

# Install dependencies
npm install

# Set up DeepSeek API key
export DEEPSEEK_API_KEY="your_api_key_here"

# Start the frontend (in one terminal)
npm run dev

# Start the backend service (in another terminal)
node server.js
```

Then open your browser to the URL shown by `npm run dev` (typically http://localhost:5173).

## Project Structure

```
PokerMind-LoRA-Tuned-LLM-for-Texas-Hold-em-Poker/
├── sft/                          # Supervised Fine-Tuning (Phase 1)
│   ├── train_poker_model.py      # SFT training script
│   └── setup.sbatch              # SLURM job configuration for H100 GPU
│
├── ppo/                          # PPO Reinforcement Learning (Phase 2)
│   ├── train_ppo.py              # PPO training script
│   ├── ppo_trainer.py            # PPO algorithm implementation
│   ├── agents.py                 # Agent wrappers
│   ├── rewards.py                # Reward calculation with GAE
│   ├── config.py                 # PPO hyperparameters
│   └── setup.sbatch              # SLURM job configuration
│
├── poker_game/                   # Texas Hold'em Game Engine
│   ├── game_logic.py             # Core poker game implementation
│   ├── game_state.py             # State representation
│   ├── hand_evaluator.py         # Hand ranking and evaluation
│   ├── deck.py                   # Card and deck management
│   └── README.md                 # Game engine documentation
│
├── frontend/                     # Interactive Web Demo (Phase 3)
│   ├── src/                      # React frontend source
│   ├── server.js                 # Backend API service
│   ├── package.json              # Node.js dependencies
│   └── README.md                 # Frontend setup guide
│
├── logs/                         # Training logs and artifacts
│   ├── sft/                      # SFT training logs
│   └── ppo/                      # PPO training logs and evaluations
│
├── poker-lora-model/             # Trained model weights
│   └── Meta-Llama-3-8B/          # LoRA adapter weights
│
├── backend/                      # Backend utilities
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

For detailed game engine documentation, see [poker_game/README.md](poker_game/README.md).

## Results & Evaluation

### Supervised Fine-Tuning (SFT)

**Training Configuration:**
- **Dataset**: 110,000 poker hands from [PokerBench](https://huggingface.co/datasets/RZ412/PokerBench) (80k postflop + 30k preflop)
- **Base Model**: [Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
- **Fine-tuning Method**: LoRA (r=16, α=32, dropout=0.1)
- **Quantization**: 4-bit NF4 with double quantization
- **Training**: 3 epochs on H100 80GB GPU (~4.6 hours)
- **Batch Size**: 128 (16 per device × 8 gradient accumulation steps)
- **Learning Rate**: 1e-6 with cosine decay

**Performance Metrics:**
- **Final Training Loss**: 0.2022 (89% improvement from initial 1.9058)
- **Train-Validation Gap**: 0.0006 (no overfitting detected ✅)
- **Accuracy by Stage**:
  - Preflop: **34.7%**
  - Postflop: **44.7%**
  - Overall: **~42%** exact-match action accuracy

**Analysis**: This level of accuracy is expected for 3 epochs of training on complex poker decision-making tasks. The model successfully learned fundamental poker patterns and strategic concepts, making it ready for enhancement through PPO reinforcement learning.

### PPO Reinforcement Learning

PPO training was completed to further improve the SFT model through self-play. Training artifacts, evaluation logs, and performance metrics can be found in `logs/ppo/`. The PPO phase uses:

- **Method**: Policy optimization with frozen reference model
- **KL Divergence Constraint**: β = 0.1 to prevent policy collapse
- **Self-Play Games**: 10,000-50,000 episodes
- **Target**: +15-20% accuracy improvement over SFT baseline

For detailed PPO results, see the evaluation logs in `logs/ppo/eval_log.jsonl` and training output in `logs/ppo/Report-*.out`.

## Hardware & Dependencies

### Hardware Requirements

**For Training:**
- **SFT**: NVIDIA H100 80GB or A100 40GB+ recommended
- **PPO**: NVIDIA H100 80GB recommended
- **Quantization**: 4-bit loading reduces memory requirements significantly

**For Inference:**
- **Minimum**: NVIDIA RTX 3090 24GB or similar
- **Demo**: Any GPU with 8GB+ VRAM

### Software Dependencies

Core dependencies include:
- **PyTorch** (≥2.0): Deep learning framework
- **Transformers** (≥4.35.0): Hugging Face model library
- **PEFT** (≥0.6.0): Parameter-Efficient Fine-Tuning
- **BitsAndBytes** (≥0.41.0): Quantization
- **Datasets** (≥2.14.0): Dataset loading and processing
- **Accelerate** (≥0.24.0): Distributed training
- **NumPy, Pandas**: Data processing
- **FastAPI, Uvicorn**: Backend services

For exact versions and complete list, see `requirements.txt`.

**Installation on PACE Cluster:**
```bash
module load anaconda3
pip install --user -r requirements.txt
```

## Technical Details

### Model Architecture
- **Base Model**: Meta-Llama-3-8B (8 billion parameters)
- **Fine-tuning**: LoRA with rank r=16, alpha=32
- **Trainable Parameters**: ~17M / 8B (0.2% of total)
- **Quantization**: 4-bit NF4 format with double quantization

### Training Optimizations
- **Mixed Precision**: bfloat16 for memory efficiency
- **Gradient Accumulation**: 8 steps to simulate larger batch sizes
- **Optimizer**: AdamW with weight decay 0.01
- **Learning Rate Schedule**: Cosine decay with warmup

### Game Engine
The custom Texas Hold'em engine ([poker_game/](poker_game/)) provides:
- ✅ Full Texas Hold'em rules (blinds, betting rounds, showdown)
- ✅ 2-10 player support
- ✅ Hand evaluation (High Card → Straight Flush)
- ✅ Action validation (FOLD/CHECK/CALL/BET/RAISE)
- ✅ Reusable for both training and demo

## Roadmap & Future Work

Potential improvements and extensions:

- [ ] **Hyperparameter Sweeps**: Systematic exploration of LoRA rank, learning rates, batch sizes
- [ ] **Larger PPO Runs**: Extended self-play training (100k+ episodes)
- [ ] **Ablation Studies**: Impact of different reward structures, KL coefficients
- [ ] **Multi-opponent Training**: Train against diverse opponent strategies
- [ ] **Tournament Mode**: Implement blind increases and multi-table tournaments
- [ ] **Explainability**: Add attention visualization and decision reasoning
- [ ] **Model Distillation**: Create smaller, faster inference models
- [ ] **Mobile Deployment**: Optimize for edge devices

## Contributing

This is a course project for CS6220 at Georgia Institute of Technology. For questions or suggestions:

1. Open an issue on GitHub
2. Contact the team via email (see Contact section)

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{pokermind2025,
  author = {bshu30},
  title = {PokerMind: LoRA-Tuned LLM for Texas Hold'em with PPO Self-Play},
  year = {2025},
  publisher = {Georgia Institute of Technology},
  journal = {CS6220 Data and Visual Analytics Course Project},
  howpublished = {\url{https://github.com/Neptunian-shushu/PokerMind-LoRA-Tuned-LLM-for-Texas-Hold-em-Poker}}
}
```

## Contact

- **GitHub**: [@Neptunian-shushu](https://github.com/Neptunian-shushu)
- **Email**: bshu30@gatech.edu
- **Issues**: [Open an issue](https://github.com/Neptunian-shushu/PokerMind-LoRA-Tuned-LLM-for-Texas-Hold-em-Poker/issues)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Links

- **Repository**: [github.com/Neptunian-shushu/PokerMind-LoRA-Tuned-LLM-for-Texas-Hold-em-Poker](https://github.com/Neptunian-shushu/PokerMind-LoRA-Tuned-LLM-for-Texas-Hold-em-Poker)
- **Dataset**: [RZ412/PokerBench on Hugging Face](https://huggingface.co/datasets/RZ412/PokerBench)
- **Base Model**: [meta-llama/Meta-Llama-3-8B on Hugging Face](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
- **Training Cluster**: PACE ICE (Georgia Tech High-Performance Computing)

---

**Project Status**: Phase 1 (SFT) ✅ | Phase 2 (PPO) ✅ | Phase 3 (Frontend Demo) ✅

