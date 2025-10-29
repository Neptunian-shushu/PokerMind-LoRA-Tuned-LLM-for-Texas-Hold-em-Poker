# ppo/ppo_trainer.py
"""
Minimal self-play trainer with on-policy REINFORCE updates over LoRA params.

- Plays one full hand with N agents (LLM or random).
- If an agent is LLMAgent, we compute differentiable label log-likelihood
  via score_candidates_train() and accumulate policy loss with terminal reward.
- Reward = (final_stack - initial_stack) / BB (see RewardCalculator).
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np

from ppo.config import PPOConfig
from ppo.rewards import RewardCalculator
from poker_game.game_logic import PokerGame
from poker_game.game_state import Action
from ppo.agents import AgentBase, RandomAgent, LLMAgent, DISCRETE_ACTIONS, _legalize_label

import torch
from torch.optim import AdamW
import torch.nn.utils as nn_utils


class PPOTrainer:
    def __init__(self, cfg: PPOConfig, reward_calculator: RewardCalculator):
        self.cfg = cfg
        self.rc = reward_calculator
        self._optim: Optional[AdamW] = None  # created lazily after seeing LLM agents

    # ----- environment & agents -----

    def _init_env(self) -> PokerGame:
        return PokerGame(
            num_players=self.cfg.num_players,
            starting_stack=self.cfg.starting_stack,
            small_blind=self.cfg.small_blind,
            big_blind=self.cfg.big_blind,
            seed=self.cfg.seed,
        )

    def _ensure_agents(self, agents: Optional[List[AgentBase]]) -> List[AgentBase]:
        assert agents is not None and len(agents) == self.cfg.num_players, \
            "Provide `agents` list matching cfg.num_players (LLM or mixed)."
        return agents

    # ----- optimizer over LoRA params (shared base supported) -----

    def _maybe_build_optimizer(self, agents: List[AgentBase]):
        if self._optim is not None:
            return
        params = []
        for ag in agents:
            if isinstance(ag, LLMAgent):
                # collect only LoRA trainable params; shared base is fine
                params += list(ag.lora_parameters())
        if params:
            self._optim = AdamW(params, lr=self.cfg.learning_rate)
        else:
            self._optim = None  # no LLM seats -> no update

    # ----- one episode with on-policy updates -----

    def play_one_episode(
        self,
        env: Optional[PokerGame] = None,
        agents: Optional[List[AgentBase]] = None
    ) -> Dict:
        env = env or self._init_env()
        agents = self._ensure_agents(agents)
        self._maybe_build_optimizer(agents)

        init_stacks = [p.stack for p in env.players]
        state = env.reset()
        done = False
        steps = 0

        # Per-agent logs for policy gradient
        # Each entry: list of (logp_chosen tensor)
        pg_logs: List[List[torch.Tensor]] = [[] for _ in range(self.cfg.num_players)]

        # Keep simple bookkeeping
        while not done and steps < self.cfg.steps_per_episode:
            current_player = state.current_player()
            pid = current_player.player_id
            legal_actions = state.get_valid_actions(current_player)

            agent = agents[pid]

            # LLM path: differentiable label selection over 10 discrete actions
            if isinstance(agent, LLMAgent) and agent.use_scoring:
                # Make sure correct adapter is active in shared-base mode
                if getattr(agent, "_controller", None) is not None:
                    agent._controller.set_adapter(agent._adapter_name)

                # Build prompt + strict tail
                prompt = state.get_llm_prompt(pid) + agent._tail_instruction()
                # logp_vec[B], prob_vec[B], choice_idx, logp_chosen
                logp_vec, prob_vec, idx, logp_chosen = agent.score_candidates_train(
                    prompt, DISCRETE_ACTIONS
                )
                label = DISCRETE_ACTIONS[int(idx)]
                action, amount = _legalize_label(label, legal_actions, state, current_player)

                # Step env
                state, hand_complete, result = env.step(action, float(amount))
                done = hand_complete
                steps += 1

                # Save per-step logp for this agent only when action was taken
                # (One decision per step by the current seat)
                # Detached advantage will be multiplied later.
                pg_logs[pid].append(logp_chosen)

            else:
                # Non-LLM or non-scoring fallback: act() without grad
                action, amount, _ = agent.act(state, legal_actions, info=None)
                state, hand_complete, result = env.step(action, float(amount))
                done = hand_complete
                steps += 1

        # Terminal rewards per player (chipEV / BB)
        final_stacks = [p.stack for p in env.players]
        terminal_rewards = []
        for pid in range(self.cfg.num_players):
            r = self.rc.calculate_reward(
                player_id=pid,
                action=Action.CHECK,        # unused in conservative scheme
                hand_result=result or {},
                initial_stack=init_stacks[pid],
                final_stack=final_stacks[pid],
                big_blind=self.cfg.big_blind
            )
            terminal_rewards.append(float(r))

        # ----- policy gradient update -----
        # Advantage: use terminal reward baseline only (no value fn). Detach to avoid leaking grads.
        if self._optim is not None:
            self._optim.zero_grad(set_to_none=True)
            total_loss = torch.tensor(0.0, device=pg_logs[0][0].device) if any(pg_logs) else None

            for pid in range(self.cfg.num_players):
                if not pg_logs[pid]:
                    continue
                adv = torch.tensor(terminal_rewards[pid], dtype=torch.float32, device=pg_logs[pid][0].device)
                # Per-step REINFORCE; here we attribute the whole terminal return to all decisions of that seat
                seat_loss = torch.tensor(0.0, device=adv.device)
                for lp in pg_logs[pid]:
                    seat_loss = seat_loss - lp * adv
                total_loss = seat_loss if total_loss is None else (total_loss + seat_loss)

            if total_loss is not None:
                nn_utils.clip_grad_norm_(self._optim.param_groups[0]['params'], self.cfg.max_grad_norm)
                total_loss.backward()
                self._optim.step()

        log = {
            "steps": steps,
            "stacks_init": init_stacks,
            "stacks_final": final_stacks,
            "result": result or {},
            "terminal_rewards": terminal_rewards,
        }
        return log