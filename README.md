# PokerMind
LoRA-fine-tuned LLM for Texas Hold'em Poker — CS6220 (Georgia Tech) course project

Overview
--------
PokerMind trains a Texas Hold'em decision model using a two-stage pipeline:
1. Supervised Fine-Tuning (SFT) of Meta-Llama-3-8B with LoRA adapters on the PokerBench dataset (expert decisions).
2. PPO Self-Play to further optimize the policy via reinforcement learning.
A web frontend (React + Vite) is included as the demo / UI entrypoint.

Highlights
----------
- Phase 1 — SFT: Complete ✅  
  - Trained on PokerBench (≈110k expert hands).
  - SFT results: Final loss = 0.2022; Preflop accuracy = 90.1%; Postflop accuracy = 71.6%; Overall ≈75.9% exact-match.
- Phase 2 — PPO: Complete ✅  
  - PPO implementation and training pipeline available under `ppo/` with logs and evaluation artifacts in `logs/ppo/`.
- Phase 3 — Frontend (demo): Complete ✅  
  - The frontend (in `frontend/`) serves as the demo.

Table of contents
-----------------
- Project overview
- Quick start
- Project structure
- Results & evaluation
- Hardware & dependencies
- Roadmap / future work
- Contributing
- Citation & contact
- License

Quick start
-----------
Prereqs:
- Python 3.10+ recommended
- Node.js (for frontend)
- GPU recommended for training (see Hardware)

Clone:
```bash
git clone https://github.com/Neptunian-shushu/PokerMind-LoRA-Tuned-LLM-for-Texas-Hold-em-Poker.git
cd PokerMind-LoRA-Tuned-LLM-for-Texas-Hold-em-Poker
```

Python environment & install:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run Supervised Fine-Tuning (SFT)
```bash
# SLURM (PACE ICE H100)
cd sft/
sbatch setup.sbatch

# Or run locally (example)
python train_poker_model.py --config configs/sft_config.yaml
```
Outputs:
- Training logs: `logs/sft/`
- Saved LoRA adapter(s): `poker-lora-model/Meta-Llama-3-8B/`

Run PPO Self-Play
```bash
cd ppo/
# Run with your chosen config
python train_ppo.py --config configs/ppo_config.yaml
```
Outputs:
- PPO logs & evaluation: `logs/ppo/`
- Checkpoints/adapters: `poker-lora-model/` (check PPO-related subfolders)

Run the frontend (demo)
```bash
cd frontend/
npm install
npm run dev

# If the frontend requires local AI service:
# export DEEPSEEK_API_KEY="YOUR_API_KEY"
# node server.js
```
See `frontend/README.md` for frontend-specific environment instructions.

Project structure
-----------------
- sft/ — Supervised fine-tuning scripts, SLURM job, configs
- ppo/ — PPO implementation (agents, trainer, configs, sbatch)
- poker_game/ — Texas Hold'em engine (game logic, state, hand evaluator)
- frontend/ — React + Vite frontend (demo UI)
- logs/ — Training and evaluation logs: `logs/sft/`, `logs/ppo/`
- poker-lora-model/ — Saved LoRA adapters & model card(s)
- requirements.txt — Python dependencies

Results & evaluation
--------------------
Supervised Fine-Tuning (SFT):
- Dataset: PokerBench (≈110k hands)
- Preflop accuracy: 90.1%
- Postflop accuracy: 71.6%
- Overall exact-match accuracy: ~75.9%

PPO Self-Play:
- PPO pipeline implemented and training/evaluation artifacts saved under `logs/ppo/`.
- See `logs/ppo/` for per-episode rewards, win rates vs frozen opponents, checkpoint histories, and evaluation summaries.

Hardware & resources
--------------------
Recommended:
- SFT: NVIDIA H100 80GB or A100 40GB+
- PPO: H100 80GB recommended for larger-scale experiments
- Inference / local demo: GPU with 8–24 GB VRAM (e.g., RTX 3090)
- Storage: 20GB+ free (checkpoint size varies)

Notes:
- LoRA adapters were used (memory-efficient fine-tuning).
- 4-bit NF4 quantization with bitsandbytes was used to reduce memory footprint where applicable.

Dependencies
------------
Install all Python dependencies:
```bash
pip install -r requirements.txt
```
Core libraries: PyTorch, Transformers, PEFT, bitsandbytes, accelerate, datasets, numpy, pandas, matplotlib. See `requirements.txt` for pinned versions.

Roadmap / future work
---------------------
Although the core pipeline (SFT → PPO → frontend) is implemented in this repository, potential next experiments:
- Larger PPO runs and hyperparameter sweeps
- Curriculum learning and opponent diversity during self-play
- Ablations on LoRA rank and quantization strategy
- Additional visualizations and web UI improvements

Contributing
------------
1. Fork the repository
2. Create a branch: `git checkout -b feat/your-feature`
3. Implement changes, add tests where relevant
4. Open a pull request describing your changes

Please open an issue first for major or breaking changes.

Citation
--------
```bibtex
@misc{pokermind2025,
  author = {bshu30},
  title = {PokerMind: LoRA-Tuned LLM for Texas Hold'em with PPO Self-Play},
  year = {2025},
  publisher = {Georgia Institute of Technology},
  journal = {CS6220 Course Project}
}
```

Contact
-------
- Issues: https://github.com/Neptunian-shushu/PokerMind-LoRA-Tuned-LLM-for-Texas-Hold-em-Poker/issues  
- Email: bshu30@gatech.edu

License
-------
MIT — see LICENSE file.
