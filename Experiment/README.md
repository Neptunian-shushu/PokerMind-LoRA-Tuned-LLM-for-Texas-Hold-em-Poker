# Poker LLM Evaluation Framework

This project evaluates three poker-playing agents in Heads-Up No-Limit Hold'em:

- **Llama-3-8B Base Model**
- **LoRA Supervised Fine-Tuned (SFT) Poker Model**
- **PPO-Trained Poker Agent (1000 episodes)**

## Features
- Run pairwise matchups between any two models
- Save full hand histories in JSONL format
- Metrics: Win Rate, Chip EV, BB/100
- Modular agent + environment design

## How to Run
Use the matchup script:

```
python -m poker.scripts.run_matchups
```

This will generate:
- `hand_history_SFT_vs_PPO.jsonl`
- `hand_history_Base_vs_PPO.jsonl`
- `hand_history_Base_vs_SFT.jsonl`

Each file contains complete hand-by-hand logs.

## Folder Structure
```
poker/
  agents/
  coordinator/
  envs/
  scripts/
  game_logic.py
  game_state.py
  hand_evaluator.py
  deck.py
```

## Output Example
```
LoRA-Poker:
  Wins: 56
  Chips Won: -20.50
  BB/100: -20.71
PPO-Poker:
  Wins: 51
  Chips Won: 20.50
  BB/100: 20.71
```

## Notes
- Models use HuggingFace + PEFT for LoRA loading
- PPO model is treated as another LoRA adapter
- All matchups automatically stop early if a player busts
