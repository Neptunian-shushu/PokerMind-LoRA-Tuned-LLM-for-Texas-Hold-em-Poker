# ppo/agents.py
from __future__ import annotations
import os, re, math, random
from typing import Dict, List, Optional, Tuple
from poker_game.game_state import Action, GameState, PlayerState

# ----- Discrete actions -----
DISCRETE_ACTIONS = [
    "fold","check","call",
    "bet_0.33pot","bet_0.50pot","bet_1.00pot","bet_allin",
    "raise_0.50pot","raise_1.00pot","raise_allin",
]
LABEL2ACTION = {
    "fold": Action.FOLD, "check": Action.CHECK, "call": Action.CALL,
    "bet_0.33pot": Action.BET, "bet_0.50pot": Action.BET, "bet_1.00pot": Action.BET, "bet_allin": Action.BET,
    "raise_0.50pot": Action.RAISE, "raise_1.00pot": Action.RAISE, "raise_allin": Action.RAISE,
}
_ACTION_ID_MAP = {Action.FOLD:0, Action.CHECK:1, Action.CALL:2, Action.BET:3, Action.RAISE:4}
def _action_to_id(a: Action) -> int: return _ACTION_ID_MAP[a]

def _amount_from_label(label: str, state: GameState, actor: PlayerState) -> float:
    """Map a discrete label to a numeric bet/raise amount."""
    pot = max(state.pot, 1e-8)
    stack = float(actor.stack)
    if label in ("bet_allin","raise_allin"): return stack
    m = re.search(r"_(\d+(?:\.\d+)?)pot$", label)
    frac = float(m.group(1)) if m else 0.0
    return max(0.0, min(pot * frac, stack))

def _legalize_label(label: str, legal_actions: List[Action], state: GameState, actor: PlayerState) -> Tuple[Action, float]:
    """Convert a target label to a legal (Action, amount); degrade to check/call/fold if needed."""
    desired = LABEL2ACTION.get(label)
    def fallback():
        if Action.CHECK in legal_actions: return Action.CHECK, 0.0
        if Action.CALL  in legal_actions: return Action.CALL,  0.0
        return Action.FOLD, 0.0
    if desired is None: return fallback()
    if desired in (Action.BET, Action.RAISE):
        if desired not in legal_actions: return fallback()
        return desired, max(0.0, _amount_from_label(label, state, actor))
    return (desired, 0.0) if desired in legal_actions else fallback()

# ----- Base agents -----
class AgentBase:
    def __init__(self, player_id: int, seed: Optional[int] = None):
        self.player_id = player_id
        if seed is not None: random.seed(seed)
    def act(self, state: GameState, legal_actions: List[Action], info: Optional[Dict]=None) -> Tuple[Action,float,Dict]:
        raise NotImplementedError
    def update(self, batch: Dict) -> Dict: return {}

class RandomAgent(AgentBase):
    def act(self, state: GameState, legal_actions: List[Action], info: Optional[Dict]=None):
        action = random.choice(legal_actions)
        amount, out = 0.0, {}
        if action in (Action.BET, Action.RAISE):
            cand = {Action.BET:["bet_0.33pot","bet_0.50pot","bet_1.00pot"], Action.RAISE:["raise_0.50pot","raise_1.00pot"]}[action]
            label = random.choice(cand); out["label"] = label
            amount = _amount_from_label(label, state, state.current_player())
        return action, float(amount), out

# ----- Optional HF/PEFT -----
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel, prepare_model_for_kbit_training
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False

# ----- Shared base + multi-adapters -----
class SharedLLMController:
    """One base model shared by seats; load/switch multiple LoRA adapters."""
    def __init__(self, base_repo_or_path: str, torch_dtype: str="float16", device_map: str="auto", trust_remote_code: bool=True, use_4bit: bool=True):
        if not _HAS_TRANSFORMERS: raise RuntimeError("transformers/peft not available.")
        self.tokenizer = AutoTokenizer.from_pretrained(base_repo_or_path, token=True, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure 4-bit quantization if requested
        quantization_config = None
        if use_4bit:
            print("Loading model with 4-bit quantization for memory efficiency...", flush=True)
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            torch_dtype = torch.float16  # Force float16 for 4-bit
        
        self.model = AutoModelForCausalLM.from_pretrained(
            base_repo_or_path, 
            device_map=device_map,
            quantization_config=quantization_config,
            torch_dtype=getattr(torch, torch_dtype) if isinstance(torch_dtype,str) else torch_dtype,
            token=True, 
            trust_remote_code=trust_remote_code, 
            low_cpu_mem_usage=True
        )
        
        # Prepare model for k-bit training if using quantization
        if use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        self.adapters_loaded: Dict[str,str] = {}
        self._peft_model = None  # Will be set when first adapter is loaded
    def load_adapter(self, adapter_path: str, name: Optional[str]=None):
        name = name or os.path.basename(adapter_path.rstrip("/"))
        if self._peft_model is None:
            # First adapter: wrap base model with PeftModel
            self._peft_model = PeftModel.from_pretrained(self.model, adapter_path, adapter_name=name)
            self.model = self._peft_model
        else:
            # Subsequent adapters: load onto existing PeftModel
            self.model.load_adapter(adapter_path, adapter_name=name)
        self.adapters_loaded[name] = adapter_path
    def set_adapter(self, name: Optional[str]): self.model.set_adapter(name)
    def device(self): return next(self.model.parameters()).device

# ----- LLM agent (A-route: candidate scoring) -----
class LLMAgent(AgentBase):
    """
    Two load modes:
      (A) shared base via SharedLLMController + adapter_name
      (B) standalone: model_path (+ optional base fallback)
    Act supports:
      - use_scoring=True: score 10 labels -> distribution -> sample/greedy
      - fallback: text generate one label and parse
    RL extensions:
      - score_candidates_train(): differentiable scoring for REINFORCE
      - lora_parameters(): iterate trainable LoRA params only
    """
    def __init__(self,
                 player_id: int,
                 controller: Optional[SharedLLMController]=None,
                 adapter_name: Optional[str]=None,
                 model_path: Optional[str]=None,
                 base_model_fallback: str="meta-llama/Meta-Llama-3-8B",
                 max_seq_len: int=512, temperature: float=0.7, top_p: float=0.9,
                 use_scoring: bool=True, device_map: str="auto", torch_dtype: Optional[str]="float16"):
        super().__init__(player_id)
        if not _HAS_TRANSFORMERS: raise RuntimeError("transformers/peft not available.")
        self.max_seq_len, self.temperature, self.top_p, self.use_scoring = max_seq_len, float(temperature), float(top_p), bool(use_scoring)
        self._controller, self._adapter_name = controller, adapter_name

        if controller is not None:
            self.tokenizer, self.model = controller.tokenizer, controller.model
            self._shared = True
        else:
            self._shared = False
            assert model_path is not None, "When controller is None, pass model_path."
            adapter_cfg = os.path.join(model_path, "adapter_config.json")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            except Exception:
                self.tokenizer = AutoTokenizer.from_pretrained(base_model_fallback, trust_remote_code=True)
            if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
            if os.path.exists(adapter_cfg):
                base = AutoModelForCausalLM.from_pretrained(
                    base_model_fallback, device_map=device_map,
                    torch_dtype=getattr(torch, torch_dtype) if isinstance(torch_dtype,str) else torch_dtype,
                    trust_remote_code=True, low_cpu_mem_usage=True
                )
                self.model = PeftModel.from_pretrained(base, model_path)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path, device_map=device_map,
                    torch_dtype=getattr(torch, torch_dtype) if isinstance(torch_dtype,str) else torch_dtype,
                    trust_remote_code=True, low_cpu_mem_usage=True
                )
        self.model.eval()
        self._gen_kwargs = dict(do_sample=True, temperature=self.temperature, top_p=self.top_p,
                                max_new_tokens=8, pad_token_id=self.tokenizer.pad_token_id,
                                eos_token_id=self.tokenizer.eos_token_id, use_cache=True)

    # ---------- prompts & parsing ----------
    @staticmethod
    def _tail_instruction() -> str:
        return ("\n\nIMPORTANT:\n"
                "Reply with exactly ONE token from this set (lowercase, no explanation): {"
                + ",".join(DISCRETE_ACTIONS) + "}\nOutput:")

    @staticmethod
    def _parse_label(text: str) -> Optional[str]:
        s = text.strip().split()[0].lower() if text.strip() else ""
        if s in DISCRETE_ACTIONS: return s
        s = s.replace("bet_1pot","bet_1.00pot").replace("raise_1pot","raise_1.00pot")
        if s in DISCRETE_ACTIONS: return s
        m = re.search(r"(fold|check|call|bet_0\.33pot|bet_0\.50pot|bet_1\.00pot|bet_allin|raise_0\.50pot|raise_1\.00pot|raise_allin)", text.lower())
        return m.group(1) if m else None

    # ---------- inference (no grad) ----------
    def _score_candidates(self, prompt: str, candidates: List[str]) -> Tuple[List[float], List[float]]:
        """Log-likelihood scoring without grad; returns (logps, probs)."""
        import torch
        device = next(self.model.parameters()).device
        with torch.no_grad():
            enc_p = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_seq_len, padding=False)
            prompt_ids = enc_p["input_ids"][0].to(device); prompt_len = prompt_ids.shape[0]
            texts = [prompt + " " + c for c in candidates]
            enc = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_seq_len)
            input_ids = enc["input_ids"].to(device); attn = enc.get("attention_mask", None)
            if attn is not None: attn = attn.to(device)
            logits = self.model(input_ids=input_ids, attention_mask=attn, use_cache=False).logits
            logprobs = torch.log_softmax(logits, dim=-1)
            logp_list = []
            for i in range(input_ids.size(0)):
                ids = input_ids[i]
                valid_len = int(attn[i].sum().item()) if attn is not None else ids.size(0)
                start = min(prompt_len, valid_len - 1)
                lp = 0.0
                for t in range(start, valid_len):
                    prev_t = t - 1; token_id = int(ids[t].item())
                    lp += float(logprobs[i, prev_t, token_id].item())
                logp_list.append(lp)
            # Numerical stability: clamp and check for invalid values
            logp_list = [max(min(lp, 100.0), -100.0) for lp in logp_list]
            if any(math.isnan(lp) or math.isinf(lp) for lp in logp_list):
                # Fallback: uniform distribution
                return [0.0]*len(candidates), [1.0/len(candidates)]*len(candidates)
            mx = max(logp_list); probs = [math.exp(lp - mx) for lp in logp_list]; s = sum(probs)
            probs = [p/s for p in probs] if s>0 else [1.0/len(candidates)]*len(candidates)
        return logp_list, probs

    # ---------- training (with grad) ----------
    def score_candidates_train(self, prompt: str, candidates: List[str]):
        """
        Differentiable scoring: returns (logp_vec[B], prob_vec[B], choice_idx, logp_chosen).
        B == len(candidates). Sequence log-likelihood only on label part.
        """
        import torch
        device = next(self.model.parameters()).device
        texts = [prompt + " " + c for c in candidates]
        enc = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_seq_len)
        input_ids = enc["input_ids"].to(device)
        attn = enc.get("attention_mask", None)
        if attn is not None: attn = attn.to(device)

        out = self.model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        logprobs = out.logits.log_softmax(dim=-1)  # [B,T,V]

        # accumulate per-seq log-likelihood (skip first token)
        logp_seq = []
        for i in range(input_ids.size(0)):
            ids = input_ids[i]
            valid_len = int(attn[i].sum().item()) if attn is not None else ids.size(0)
            lp = torch.zeros((), device=device)
            for t in range(1, valid_len):
                lp = lp + logprobs[i, t-1, ids[t]]
            logp_seq.append(lp)
        logp = torch.stack(logp_seq, dim=0)        # [B]
        
        # Numerical stability: clamp logp to prevent overflow/underflow in softmax
        logp = torch.clamp(logp, min=-100.0, max=100.0)
        
        # Check for NaN/inf before softmax
        if torch.isnan(logp).any() or torch.isinf(logp).any():
            # Fallback: uniform distribution
            print(f"WARNING: Invalid logp detected, using uniform distribution", flush=True)
            logp = torch.zeros_like(logp)
        
        probs = torch.softmax(logp, dim=-1)        # [B]
        
        # Additional safety: ensure probs are valid for multinomial
        if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
            print(f"WARNING: Invalid probs after softmax, using uniform", flush=True)
            probs = torch.ones_like(probs) / probs.size(0)

        # sample/greedy on label set
        if self.temperature > 0:
            idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            idx = torch.argmax(probs, dim=-1)
        logp_chosen = logp.gather(0, idx.view(-1)).squeeze()
        return logp, probs, int(idx.item()), logp_chosen

    def lora_parameters(self):
        """Yield trainable LoRA parameters only."""
        for n, p in self.model.named_parameters():
            if "lora_" in n and p.requires_grad:
                yield p

    # ---------- act ----------
    def act(self, state: GameState, legal_actions: List[Action], info: Optional[Dict]=None) -> Tuple[Action,float,Dict]:
        if self._controller is not None: self._controller.set_adapter(self._adapter_name)
        prompt = state.get_llm_prompt(self.player_id)

        if self.use_scoring:
            full_prompt = prompt + self._tail_instruction()
            _, probs = self._score_candidates(full_prompt, DISCRETE_ACTIONS)
            import torch as _t
            if self.temperature > 0:
                idx = int(_t.multinomial(_t.tensor(probs), 1).item())
            else:
                idx = int(_t.argmax(_t.tensor(probs)).item())
            label = DISCRETE_ACTIONS[idx]
            actor = state.current_player()
            action, amount = _legalize_label(label, legal_actions, state, actor)
            return action, float(amount), {"label": label, "probs": probs}

        full_prompt = prompt + self._tail_instruction()
        inputs = self.tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=self.max_seq_len, padding=False)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(**inputs, do_sample=True, temperature=self.temperature,
                                      top_p=self.top_p, max_new_tokens=8,
                                      pad_token_id=self.tokenizer.pad_token_id,
                                      eos_token_id=self.tokenizer.eos_token_id, use_cache=True)
        full = self.tokenizer.decode(out[0], skip_special_tokens=True)
        gen = full[len(full_prompt):].strip()
        label = self._parse_label(gen) or "fold"
        actor = state.current_player()
        action, amount = _legalize_label(label, legal_actions, state, actor)
        return action, float(amount), {"label": label, "raw": gen}
    
    def policy_probs_for_prompt(self, prompt: str):
        full_prompt = prompt + self._tail_instruction()
        _, probs = self._score_candidates(full_prompt, DISCRETE_ACTIONS)
        return probs