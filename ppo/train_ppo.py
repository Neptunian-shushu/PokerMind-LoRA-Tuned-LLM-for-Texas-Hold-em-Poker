# ppo/train_ppo.py
"""
Self-play RL entrypoint (A-route: shared base + LoRA adapters).
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
def _format_pb_prompt(inst: str) -> str:
    return f"### Instruction:\n{inst}\n\n### Response:\n"

def _tail_instruction() -> str:
    return (
        "\n\nIMPORTANT:\n"
        "Reply with exactly ONE token from this set (lowercase, no explanation): {"
        + ",".join(DISCRETE_ACTIONS)
        + "}\nOutput:"
    )

def _parse_pb_target_to_label(text: str) -> str:
    t = (text or "").strip().lower().split()
    if not t:
        return "fold"
    a = t[0]
    if a in ("fold", "check", "call"):
        return a
    if a == "bet":
        return "bet_0.50pot"
    if a == "raise":
        return "raise_1.00pot"
    return "fold"

@torch.no_grad()
def _score_candidates(tokenizer, model, prompt: str, candidates):
    device = next(model.parameters()).device
    enc_p = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding=False)
    prompt_ids = enc_p["input_ids"][0].to(device)
    prompt_len = prompt_ids.shape[0]
    texts = [prompt + " " + c for c in candidates]
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask", None)
    if attn is not None:
        attn = attn.to(device)
    logits = model(input_ids=input_ids, attention_mask=attn, use_cache=False).logits
    logprobs = torch.log_softmax(logits, dim=-1)
    logp = []
    for i in range(input_ids.size(0)):
        ids = input_ids[i]
        valid = int(attn[i].sum().item()) if attn is not None else ids.size(0)
        start = min(prompt_len, valid - 1)
        s = 0.0
        for t in range(start, valid):
            prev = t - 1
            tok = int(ids[t].item())
            s += float(logprobs[i, prev, tok].item())
        logp.append(s)
    mx = max(logp)
    probs = [np.exp(v - mx) for v in logp]
    s = float(sum(probs))
    probs = [p / s for p in probs] if s > 0 else [1.0 / len(candidates)] * len(candidates)
    return probs

def eval_pokerbench_accuracy(ctl: SharedLLMController, adapter_name: str, max_samples: int = 1000) -> float:
    try:
        from datasets import load_dataset
    except Exception:
        print("[eval] datasets not available; skip.")
        return -1.0

    # switch adapter
    ctl.set_adapter(adapter_name)
    tok = ctl.tokenizer
    mdl = ctl.model.eval()

    ds = load_dataset("RZ412/PokerBench")
    test = ds["test"]
    n = min(max_samples, len(test))
    correct = 0
    total = 0

    for i in range(n):
        inst = test[i]["instruction"]
        out = test[i]["output"]
        target_label = _parse_pb_target_to_label(out)
        prompt = _format_pb_prompt(inst) + _tail_instruction()
        probs = _score_candidates(tok, mdl, prompt, DISCRETE_ACTIONS)
        pred = DISCRETE_ACTIONS[int(np.argmax(probs))]
        # collapse bet/raise families to action verb for fairness
        def coarse(lbl):
            if lbl.startswith("bet_"): return "bet"
            if lbl.startswith("raise_"): return "raise"
            return lbl
        if coarse(pred) == coarse(target_label):
            correct += 1
        total += 1

    acc = (correct / total) * 100.0 if total > 0 else -1.0
    print(f"[eval] PokerBench action-accuracy (coarse) on {total} samples = {acc:.2f}%")
    return acc

# ---- main loop ----
def main(cfg: PPOConfig = DEFAULT_CONFIG):
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(getattr(cfg, "rl_adapter_save_dir", os.path.join(cfg.output_dir, "rl_adapters")), exist_ok=True)
    set_global_seeds(cfg.seed)

    rc = RewardCalculator(big_blind=cfg.big_blind)
    trainer = PPOTrainer(cfg, reward_calculator=rc)
    agents, ctl = build_agents_and_controller(cfg)

    t0 = time.time()
    for ep in range(1, cfg.num_episodes + 1):
        log = trainer.play_one_episode(agents=agents)

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
            acc = eval_pokerbench_accuracy(ctl, name, max_samples=min(1000, getattr(cfg, "eval_episodes", 100)))
            # append a tiny eval log
            eval_log_path = os.path.join(cfg.log_dir, "eval_log.jsonl")
            with open(eval_log_path, "a") as wf:
                wf.write(json.dumps({"episode": ep, "adapter": name, "pokerbench_action_acc": acc, "time": time.time()}) + "\n")

    dt = time.time() - t0
    print(f"Done. Episodes={cfg.num_episodes}, elapsed={dt:.1f}s")

if __name__ == "__main__":
    main()