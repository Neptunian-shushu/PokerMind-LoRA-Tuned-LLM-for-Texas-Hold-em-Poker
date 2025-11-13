# ppo/train_ppo.py
"""
Self-play RL entrypoint (Shared base + LoRA adapters).
"""

from __future__ import annotations
import os, json, time, random
import numpy as np
import torch

from ppo.config import DEFAULT_CONFIG, PPOConfig
from ppo.rewards import RewardCalculator
from ppo.ppo_trainer import PPOTrainer
from ppo.agents import LLMAgent, SharedLLMController, DISCRETE_ACTIONS

# ---- build agents + controller ----
def build_agents_and_controller(cfg: PPOConfig):
    ctl = SharedLLMController(
        base_repo_or_path=cfg.base_repo_or_path,
        torch_dtype=cfg.torch_dtype,
        device_map=cfg.device_map,
        trust_remote_code=True,
        use_4bit=False,
    )
    for path, name in zip(cfg.adapter_paths, cfg.adapter_register_names):
        ctl.load_adapter(path, name=name)

    agents = []
    for pid in range(cfg.num_players):
        agents.append(
            LLMAgent(
                player_id=pid,
                controller=ctl,
                adapter_name=cfg.seat_adapter_names[pid],
                max_seq_len=cfg.max_seq_len,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                use_scoring=cfg.use_scoring,
            )
        )
    return agents, ctl

def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ---- evaluation: PokerBench accuracy (action label match) ----
def _format_prompt(instruction: str) -> str:
    """Format prompt to match SFT training format: ### Instruction / ### Response"""
    return f"### Instruction:\n{instruction}\n\n### Response:\n"

def extract_action_and_amount(output_text):
    """
    Extract action and amount from model output (matching SFT evaluation).
    Returns: (action, amount) tuple
    """
    tokens = output_text.strip().lower().split()
    if not tokens:
        return None, None
    action = tokens[0]
    amount = None
    if action in ['bet', 'raise', 'call'] and len(tokens) > 1:
        try:
            amount = float(tokens[1])
        except (ValueError, IndexError):
            pass
    return action, amount

def eval_pokerbench_accuracy(ctl: SharedLLMController, adapter_name: str, max_preflop: int = 100, max_postflop: int = 1000) -> dict:
    """
    Evaluate model accuracy using generation (matching SFT evaluation method).
    PokerBench test split has ~999 preflop and ~10001 postflop samples:
    - Postflop: indices 0-9999 (we'll use first max_postflop)
    - Preflop: indices 10000-10998 (we'll use first max_preflop)
    
    Returns dict with detailed metrics for logging.
    """
    try:
        from datasets import load_dataset
    except Exception:
        print("[eval] datasets not available; skip.")
        return {"overall_acc": -1.0}

    # switch adapter
    ctl.set_adapter(adapter_name)
    tok = ctl.tokenizer
    mdl = ctl.model.eval()

    ds = load_dataset("RZ412/PokerBench")
    # Use TEST split
    test = ds["test"]

    # Postflop: first max_postflop samples (indices 0 to max_postflop-1)
    postflop_indices = list(range(0, min(max_postflop, 10000, len(test))))
    # Preflop: from index 10000 onwards, take first max_preflop
    preflop_start = 10000
    preflop_end = min(preflop_start + max_preflop, len(test))
    preflop_indices = list(range(preflop_start, preflop_end)) if preflop_start < len(test) else []
    
    # Helper parsers to match SFT behavior
    def _parse_pb_target_to_label(target_text: str):
        # Return action string (e.g., 'fold','call','bet','raise','check')
        a, _ = extract_action_and_amount(target_text)
        return a

    def _parse_generated_action(gen_text: str):
        a, _ = extract_action_and_amount(gen_text)
        return a

    # Evaluate both categories separately
    def eval_subset(indices, name):
        n = len(indices)
        action_correct = 0
        exact_match_correct = 0
        device = next(mdl.parameters()).device
        
        for i in range(n):
            idx = indices[i]
            inst = test[idx]["instruction"]
            out = test[idx]["output"]
            
            # Get expected action and amount (SFT-style)
            expected_action, expected_amount = extract_action_and_amount(out)

            # Format prompt (matching SFT training format)
            prompt = _format_prompt(inst)
            
            try:
                # Generate response using same settings as SFT evaluation
                inputs = tok(prompt, return_tensors="pt", return_attention_mask=True).to(device)
                with torch.no_grad():
                    outputs = mdl.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_length=inputs['input_ids'].shape[1] + 50,
                        temperature=0.1,
                        do_sample=True,
                        top_p=0.95,
                        pad_token_id=tok.pad_token_id,
                        eos_token_id=tok.eos_token_id,
                        use_cache=True
                    )
                
                response = tok.decode(outputs[0], skip_special_tokens=True)
                generated_part = response[len(prompt):].strip()
                
                # Extract generated action and amount
                gen_action, gen_amount = extract_action_and_amount(generated_part)

                # Compare actions (action-level accuracy)
                if expected_action is not None and gen_action == expected_action:
                    action_correct += 1
                    # Check exact match: for bet/raise/call verify amounts, for fold/check just action match is exact
                    if expected_action in ['bet', 'raise', 'call']:
                        if expected_amount is not None and gen_amount is not None:
                            if abs(expected_amount - gen_amount) < 0.01:
                                exact_match_correct += 1
                        elif expected_amount is None and gen_amount is None:
                            exact_match_correct += 1
                    else:
                        # For fold/check, action match = exact match (no amount to verify)
                        exact_match_correct += 1
                    
            except Exception as e:
                # On error, count as incorrect
                continue
        
        acc = (action_correct / n) * 100.0 if n > 0 else -1.0
        exact_pct = (exact_match_correct / n) * 100.0 if n > 0 else 0.0
        print(f"[eval] {name} action-accuracy: {acc:.2f}% ({action_correct}/{n} samples), exact-match: {exact_pct:.2f}% ({exact_match_correct}/{n})")
        return acc, exact_pct, action_correct, exact_match_correct, n
    
    preflop_acc, preflop_exact, preflop_correct, preflop_exact_correct, preflop_n = eval_subset(preflop_indices, "Preflop")
    postflop_acc, postflop_exact, postflop_correct, postflop_exact_correct, postflop_n = eval_subset(postflop_indices, "Postflop")
    
    # Overall accuracy
    total_correct = preflop_correct + postflop_correct
    total_n = preflop_n + postflop_n
    overall_acc = (total_correct / total_n) * 100.0 if total_n > 0 else -1.0
    print(f"[eval] Overall accuracy: {overall_acc:.2f}% ({total_correct}/{total_n} samples)")
    
    # Return detailed metrics as dict for logging
    return {
        "overall_acc": overall_acc,
        "overall_correct": total_correct,
        "overall_total": total_n,
        "preflop_action_acc": preflop_acc,
        "preflop_action_correct": preflop_correct,
        "preflop_exact_match": preflop_exact,
        "preflop_exact_correct": preflop_exact_correct,
        "preflop_total": preflop_n,
        "postflop_action_acc": postflop_acc,
        "postflop_action_correct": postflop_correct,
        "postflop_exact_match": postflop_exact,
        "postflop_exact_correct": postflop_exact_correct,
        "postflop_total": postflop_n,
    }

# ---- main loop ----
def main(cfg: PPOConfig = DEFAULT_CONFIG):
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(getattr(cfg, "rl_adapter_save_dir", os.path.join(cfg.output_dir, "rl_adapters")), exist_ok=True)
    set_global_seeds(cfg.seed)

    rc = RewardCalculator(big_blind=cfg.big_blind)
    trainer = PPOTrainer(cfg, reward_calculator=rc)
    agents, ctl = build_agents_and_controller(cfg)

    # Baseline evaluation before any PPO updates (to match SFT baseline)
    try:
        name0 = cfg.seat_adapter_names[0]
        print("[eval] Baseline (episode 0) PokerBench accuracy:")
        eval_results = eval_pokerbench_accuracy(ctl, name0, max_preflop=100, max_postflop=1000)
        # Log baseline results
        eval_log_path = os.path.join(cfg.log_dir, "eval_log.jsonl")
        with open(eval_log_path, "a") as wf:
            wf.write(json.dumps({"episode": 0, "adapter": name0, **eval_results, "time": time.time()}) + "\n")
    except Exception as e:
        print(f"[eval] Baseline eval skipped due to error: {e}")

    t0 = time.time()
    print(f"Starting training loop: {cfg.num_episodes} episodes", flush=True)
    for ep in range(1, cfg.num_episodes + 1):
        print(f"Episode {ep} starting...", flush=True)
        log = trainer.play_one_episode(agents=agents)
        print(f"Episode {ep} completed: {log.get('steps', 0)} steps", flush=True)

        if ep % cfg.log_frequency == 0:
            tr = log.get("terminal_rewards", [])
            print(f"[{ep:6d}] steps={log['steps']:2d} stacks={log['stacks_init']}→{log['stacks_final']} termR={['%+.3f'%x for x in tr]}")

        if cfg.save_frequency and (ep % cfg.save_frequency == 0):
            ckpt_path = os.path.join(cfg.output_dir, f"ckpt_ep{ep}.json")
            with open(ckpt_path, "w") as f:
                json.dump({"episode": ep, "config": cfg.to_dict(), "last_log": log}, f, indent=2)
            print(f"Saved checkpoint: {ckpt_path}")

        # save LoRA adapters
        save_every = getattr(cfg, "save_adapter_every", 0)
        if save_every and (ep % save_every == 0):
            tag = f"ep{ep:06d}"
            out_dir = os.path.join(getattr(cfg, "rl_adapter_save_dir", os.path.join(cfg.output_dir, "rl_adapters")), tag)
            os.makedirs(out_dir, exist_ok=True)
            ctl.model.save_pretrained(out_dir)
            ctl.tokenizer.save_pretrained(out_dir)
            print(f"[save] LoRA adapters saved to {out_dir}")

        # eval PokerBench
        if cfg.eval_frequency and (ep % cfg.eval_frequency == 0):
            # use first seat's adapter for eval
            name = cfg.seat_adapter_names[0]
            eval_results = eval_pokerbench_accuracy(ctl, name, max_preflop=100, max_postflop=1000)
            # append detailed eval log
            eval_log_path = os.path.join(cfg.log_dir, "eval_log.jsonl")
            with open(eval_log_path, "a") as wf:
                wf.write(json.dumps({"episode": ep, "adapter": name, **eval_results, "time": time.time()}) + "\n")

    dt = time.time() - t0
    print(f"Done. Episodes={cfg.num_episodes}, elapsed={dt:.1f}s")

if __name__ == "__main__":
    # Use FAST_CONFIG for testing (100 episodes), or DEFAULT_CONFIG for full training (10,000 episodes)
    from ppo.config import FAST_CONFIG
    main(cfg=FAST_CONFIG)