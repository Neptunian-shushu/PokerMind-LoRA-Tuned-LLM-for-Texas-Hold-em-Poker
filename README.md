# PokerMind: LoRA-Tuned LLM for Texas Hold'em Poker# PokerMind: LoRA-Tuned LLM for Texas Hold'em Poker



CS6220 Data and Visual Analytics Project - Georgia Institute of TechnologyThis repository contains the CS6220 (Data and Visual Analytics) project implementing LoRA fine-tuning of large language models for Texas Hold'em poker decision making, enhanced with PPO (Proximal Policy Optimization) self-play reinforcement learning.



A two-phase approach to training AI poker agents: **Supervised Fine-Tuning (SFT)** followed by **PPO Self-Play** reinforcement learning.## Project Overview



---This project uses a two-phase approach to train AI poker agents:

1. **Supervised Fine-Tuning (SFT)**: Fine-tune Llama-3-8B using LoRA on the PokerBench dataset (110k expert poker decisions)

## 🎯 Project Overview2. **PPO Self-Play**: Further improve the model through reinforcement learning via self-play against itself



This project fine-tunes Meta-Llama-3-8B on poker decision-making using:The project achieves **35-45% action accuracy** after SFT and targets **55-65%+** accuracy after PPO training.



1. **Phase 1 - SFT**: Train on 110k expert poker hands from PokerBench dataset## Quick Start

   - ✅ **Complete**: 35-45% action accuracy achieved

   ### Phase 1: Supervised Fine-Tuning (✅ COMPLETE)

2. **Phase 2 - PPO**: Improve through self-play reinforcement learning  ```bash

   - 🚧 **In Progress**: Target 55-65%+ accuracycd sft/

sbatch setup.sbatch  # Run on PACE ICE H100 GPU

---```



## 📁 Project Structure**Results**: 35-45% action accuracy, model saved to `poker-lora-model/`



```### Phase 2: PPO Self-Play (🚧 IN PROGRESS)

CS6220/```bash

├── sft/                          # Supervised Fine-Tuning (✅ Complete)cd ppo/

│   ├── train_poker_model.py      # SFT training scriptpython train_ppo.py  # After implementation complete

│   └── setup.sbatch               # SLURM job for H100 GPU```

│

├── ppo/                          # PPO Reinforcement Learning (🚧 Framework Ready)### Demo Interface (⏳ PLANNED)

│   ├── config.py                  # PPO hyperparameters```bash

│   ├── rewards.py                 # Reward calculation with GAEcd demo/

│   ├── agents.py                  # Agent wrappers (TODO)python app.py  # Play against trained AI

│   ├── ppo_trainer.py             # PPO algorithm (TODO)```

│   └── train_ppo.py               # Main PPO training loop (TODO)

│## Project Structure

├── poker_game/                   # Texas Hold'em Game Engine (✅ Complete)

│   ├── game_logic.py              # Core poker game```

│   ├── game_state.py              # State representationCS6220/

│   ├── hand_evaluator.py          # Hand ranking├── sft/                          # Supervised Fine-Tuning

│   ├── deck.py                    # Card management│   ├── train_poker_model.py      # SFT training script

│   └── README.md                  # Game engine documentation│   └── setup.sbatch               # SLURM job for H100 GPU

││

├── demo/                         # Interactive Demo (⏳ Planned)├── ppo/                          # PPO Reinforcement Learning

│   └── app.py                     # Gradio web interface│   ├── config.py                  # PPO hyperparameters

││   ├── rewards.py                 # Reward calculation

├── logs/                         # Training logs│   ├── agents.py                  # Agent wrappers (TODO)

│   ├── sft/                       # SFT training outputs│   ├── ppo_trainer.py             # PPO algorithm (TODO)

│   └── ppo/                       # PPO training outputs│   └── train_ppo.py               # Main PPO script (TODO)

││

├── poker-lora-model/             # Trained Models├── poker_game/                   # Reusable Game Engine

│   └── Meta-Llama-3-8B/          # SFT LoRA adapter weights│   ├── game_logic.py              # Core poker game

││   ├── game_state.py              # State representation

└── requirements.txt              # Python dependencies│   ├── hand_evaluator.py          # Hand ranking

```│   └── deck.py                    # Card management

│

---├── demo/                         # Interactive Demo (TODO)

│   └── app.py                     # Gradio interface

## 🚀 Quick Start│

├── utils/                        # Shared utilities

### Phase 1: Supervised Fine-Tuning (✅ Complete)│   └── (utility scripts)

│

```bash├── logs/                         # Training logs

cd sft/│   ├── sft/                       # SFT logs

sbatch setup.sbatch  # Run on PACE ICE H100 GPU│   └── ppo/                       # PPO logs

```│

├── models/                       # Model checkpoints

**Results:**└── poker-lora-model/             # Trained SFT model

- Preflop accuracy: 34.7%    └── Meta-Llama-3-8B/          # LoRA adapter weights

- Postflop accuracy: 44.7%

- Model saved to `poker-lora-model/Meta-Llama-3-8B/````



### Phase 2: PPO Self-Play (🚧 In Progress)See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed documentation.



```bash## Package Requirements

cd ppo/

python train_ppo.py  # After implementation completeCore dependencies include:

```- **PyTorch, Transformers, PEFT**: ML frameworks and fine-tuning

- **NumPy, Pandas, Matplotlib**: Data analysis and visualization

### Demo Interface (⏳ Planned)- **Jupyter, Datasets, BitsAndBytes**: Development and optimization



```bashSee `requirements.txt` for complete list and `SETUP.md` for detailed installation guide.

cd demo/

python app.py  # Play against trained AI## Hardware Requirements

```

- **Minimum**: 8GB RAM, multi-core CPU

---- **Recommended**: 16GB+ RAM, NVIDIA GPU with 8GB+ VRAM

- **Storage**: 20GB+ free space

## 📊 Training Results

## Team Setup

### Supervised Fine-Tuning (SFT) - COMPLETE ✅

For teammates to get started:

**Configuration:**1. Clone this repository

- Dataset: 110,000 poker hands (80k postflop + 30k preflop)2. Follow instructions in `SETUP.md`

- Model: Meta-Llama-3-8B with LoRA (r=16, α=32)3. Install packages from `requirements.txt`

- Training: 3 epochs on H100 80GB GPU4. Open and run `poker_finetuning.ipynb`

- Duration: ~4.6 hours

- Batch size: 128 (16 × 8 accumulation)## Training Results



**Results:**### Supervised Fine-Tuning (SFT)

- Final loss: **0.2022** (89% improvement from 1.9058)- **Dataset**: 110,000 poker hands (80k postflop + 30k preflop)

- Train-val gap: **0.0006** (no overfitting ✅)- **Training**: 3 epochs on H100 GPU (~4.6 hours)

- **Preflop**: 34.7% action accuracy- **Final Loss**: 0.2022 (89% improvement from 1.9058)

- **Postflop**: 44.7% action accuracy  - **Accuracy**: 

- **Overall**: ~42% exact match accuracy  - Preflop: 34.7%

  - Postflop: 44.7%

**Analysis:** This accuracy is expected for 3 epochs on complex poker decision-making. The model learned fundamental patterns and is ready for PPO enhancement.  - Overall: ~42% exact match



### PPO Self-Play - PLANNED 🚧**Analysis**: This accuracy is expected and normal for 3 epochs of training on complex poker decision-making. The model has learned fundamental poker patterns and is ready for PPO enhancement.



**Method:**### PPO Self-Play (Planned)

- Policy optimization with frozen reference model- **Method**: Policy optimization with frozen reference model

- KL divergence constraint (β = 0.1)- **Expected Improvement**: +15-20% accuracy (target 55-65%)

- 10,000-50,000 self-play games- **Training**: 10,000-50,000 self-play games

- **Key Innovation**: KL divergence constraint to prevent policy collapse

**Expected Results:**

- Target accuracy: **55-65%+** (+15-20% improvement)## Key Features

- Better strategic play (aggression, bluffing, adaptation)

- Win rate vs frozen opponent: 60%+✅ **LoRA Fine-Tuning**: Memory-efficient training (only 0.2% params trainable)  

✅ **4-bit Quantization**: Fits 8B model on single GPU  

---✅ **Complete Game Engine**: Reusable Texas Hold'em implementation  

✅ **Modular Architecture**: Separate modules for SFT, PPO, game logic, demo  

## 🎮 Poker Game Engine🚧 **PPO Self-Play**: In development  

⏳ **Interactive Demo**: Planned Gradio interface  

Complete Texas Hold'em implementation for PPO training and demo interface.

## Contributing

**Features:**

- ✅ Full game rules (blinds, betting, showdown)This is a course project for CS6220 at Georgia Tech. Team members should coordinate through the established communication channels.
- ✅ 2-10 player support
- ✅ Hand evaluation (High Card → Straight Flush)
- ✅ Action validation (FOLD/CHECK/CALL/BET/RAISE)
- ✅ State tracking and history
- ✅ Reusable for training and demo

**Example:**
```python
from poker_game import PokerGame, Action

game = PokerGame(num_players=2, starting_stack=200.0)
state = game.reset()

# Get valid actions and execute
valid_actions = state.get_valid_actions(state.current_player())
state, done, result = game.step(Action.RAISE, amount=6.0)

if done:
    print(f"Winners: Player {result['winners']}")
```

See [`poker_game/README.md`](poker_game/README.md) for full documentation.

---

## 🤖 PPO Framework

### Current Status

**Complete:**
- ✅ `config.py` - Hyperparameters (γ=0.99, ε=0.2, KL=0.1)
- ✅ `rewards.py` - Reward calculation with GAE

**TODO:**
- ⏳ `agents.py` - Model wrappers (policy + reference)
- ⏳ `ppo_trainer.py` - PPO algorithm implementation
- ⏳ `train_ppo.py` - Training loop with self-play

### Reward Structure

```python
Win at showdown:  +1.0 + stack_change
Win by fold:      +1.0 + stack_change  
Lose:             -1.0 + stack_change
Fold penalty:     -0.1 (encourage playing)
Showdown bonus:   +0.2 (reward good play)
```

### Configuration

```python
from ppo.config import PPOConfig, PRODUCTION_CONFIG

# Production settings
config = PRODUCTION_CONFIG
# - 50k episodes
# - Batch size 64
# - Learning rate 5e-7
# - KL coefficient 0.1
```

---

## 🔧 Technical Details

### Model Architecture
- **Base Model**: Meta-Llama-3-8B
- **Fine-tuning**: LoRA (r=16, alpha=32, dropout=0.1)
- **Quantization**: 4-bit NF4 with double quantization
- **Trainable Parameters**: ~17M / 8B (0.2%)

### Training Configuration

**SFT Phase:**
- Batch size: 128 (16 per device × 8 accumulation)
- Learning rate: 1e-6 with cosine decay
- Optimizer: AdamW (weight_decay=0.01)
- Hardware: NVIDIA H100 80GB HBM3

**PPO Phase:**
- Episodes: 10,000-50,000 self-play games
- KL coefficient: 0.1 (stay close to SFT policy)
- Clip epsilon: 0.2 (standard PPO)
- Discount factor: 0.99
- GAE lambda: 0.95

---

## 💻 Hardware Requirements

- **SFT Training**: H100 80GB or A100 40GB+
- **PPO Training**: H100 80GB recommended
- **Inference**: RTX 3090 24GB or similar
- **Demo**: Any GPU with 8GB+ VRAM

---

## 📦 Dependencies

```bash
pip install -r requirements.txt
```

**Core packages:**
- PyTorch, Transformers, PEFT
- BitsAndBytes (quantization)
- Datasets, NumPy, Pandas
- Accelerate (distributed training)

---

## 🎯 Next Steps

1. **Implement PPO Components**
   - [ ] Agent wrappers (`ppo/agents.py`)
   - [ ] PPO trainer (`ppo/ppo_trainer.py`)
   - [ ] Training script (`ppo/train_ppo.py`)

2. **Run PPO Training**
   - [ ] Test with 1k episodes
   - [ ] Full training with 10k-50k episodes
   - [ ] Evaluate vs frozen SFT opponent

3. **Build Demo Interface**
   - [ ] Gradio web UI
   - [ ] Human vs AI matches
   - [ ] Game state visualization

4. **Final Evaluation**
   - [ ] Test on held-out scenarios
   - [ ] Compare SFT vs PPO performance
   - [ ] Analyze learned strategies

---

## 📚 Documentation

- **Main README**: This file - project overview and setup
- **[poker_game/README.md](poker_game/README.md)**: Game engine API and examples
- **[logs/sft/](logs/sft/)**: SFT training logs and analysis

---

## 🔗 Links

- **Repository**: [github.com/Neptunian-shushu/PokerMind-LoRA-Tuned-LLM-for-Texas-Hold-em-Poker](https://github.com/Neptunian-shushu/PokerMind-LoRA-Tuned-LLM-for-Texas-Hold-em-Poker)
- **Dataset**: [RZ412/PokerBench](https://huggingface.co/datasets/RZ412/PokerBench)
- **Base Model**: [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
- **Cluster**: PACE ICE (Georgia Tech)

---

## 📝 Citation

```bibtex
@misc{pokermind2025,
  author = {bshu30},
  title = {PokerMind: LoRA-Tuned LLM for Texas Hold'em with PPO Self-Play},
  year = {2025},
  publisher = {Georgia Institute of Technology},
  journal = {CS6220 Course Project}
}
```

---

## 📧 Contact

- **GitHub Issues**: Open an issue on the repository
- **Email**: bshu30@gatech.edu

---

## 📄 License

MIT License - See LICENSE file for details

---

**Status**: Phase 1 (SFT) ✅ | Phase 2 (PPO) 🚧 | Phase 3 (Demo) ⏳
