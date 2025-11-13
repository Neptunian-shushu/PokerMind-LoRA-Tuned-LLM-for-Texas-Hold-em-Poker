# ppo/ppo_trainer.py
"""
Self-play trainer with proper PPO (Proximal Policy Optimization) implementation.

Key PPO components implemented:
1. Value network for baseline/advantage estimation
2. Experience buffer for storing trajectories
3. GAE (Generalized Advantage Estimation) for better advantage calculation
4. Clipped surrogate objective to limit policy updates
5. Multiple epochs over collected data for sample efficiency
6. Entropy bonus to encourage exploration
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np

from ppo.config import PPOConfig
from ppo.rewards import RewardCalculator
from poker_game.game_logic import PokerGame
from poker_game.game_state import Action, GameState
from ppo.agents import AgentBase, RandomAgent, LLMAgent, DISCRETE_ACTIONS, _legalize_label

import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.utils as nn_utils


class ValueNetwork(nn.Module):
    """
    MLP value function that estimates V(s) for a given game state.
    Input: numerical features extracted from game state
    Output: scalar state value estimate
    """
    def __init__(self, input_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        """x: [batch_size, input_dim] -> [batch_size]"""
        return self.net(x).squeeze(-1)


class TrajectoryBuffer:
    """
    Stores trajectories from multiple episodes for batch PPO updates.
    Each trajectory contains: state_features, log_probs, rewards, values, dones
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Clear all stored trajectories"""
        self.state_features = []  # List of state feature tensors
        self.log_probs = []       # List of action log probabilities (detached)
        self.rewards = []         # List of rewards
        self.values = []          # List of value estimates V(s)
        self.dones = []           # List of done flags
        self.player_ids = []      # Track which player took each action
        self.prompts = []         # List of prompt strings (for recomputing log-probs)
        self.action_idxs = []     # List of chosen action indices (int)
    
    def add(self, state_feat, log_prob, reward, value, done, player_id):
        """Add a transition to the buffer"""
        self.state_features.append(state_feat)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.player_ids.append(player_id)

    def add_with_prompt(self, state_feat, log_prob, reward, value, done, player_id, prompt: str, action_idx: int):
        """Add transition including prompt and chosen action index."""
        # Ensure stored log_prob is detached (no graph retained)
        if isinstance(log_prob, torch.Tensor):
            self.log_probs.append(log_prob.detach())
        else:
            # Convert numeric to tensor
            try:
                self.log_probs.append(torch.tensor(float(log_prob)))
            except Exception:
                self.log_probs.append(torch.tensor(0.0))
        self.state_features.append(state_feat)
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.dones.append(float(done))
        self.player_ids.append(int(player_id))
        self.prompts.append(str(prompt))
        self.action_idxs.append(int(action_idx))
    
    def size(self):
        """Return number of transitions stored"""
        return len(self.rewards)
    
    def get_tensors(self, device):
        """Convert lists to tensors for batch processing"""
        return {
            'state_features': torch.stack(self.state_features).to(device),
            'log_probs': torch.stack(self.log_probs).to(device),
            'rewards': torch.tensor(self.rewards, dtype=torch.float32, device=device),
            'values': torch.tensor(self.values, dtype=torch.float32, device=device),
            'dones': torch.tensor(self.dones, dtype=torch.float32, device=device),
            'player_ids': torch.tensor(self.player_ids, dtype=torch.long, device=device),
            'prompts': list(self.prompts),
            'action_idxs': torch.tensor(self.action_idxs, dtype=torch.long, device=device),
        }


class PPOTrainer:
    def __init__(self, cfg: PPOConfig, reward_calculator: RewardCalculator):
        self.cfg = cfg
        self.rc = reward_calculator
        self._policy_optim: Optional[AdamW] = None  # optimizer for policy (LoRA params)
        self._value_optim: Optional[AdamW] = None   # optimizer for value function
        
        # Value network for each player
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.value_nets = [ValueNetwork().to(device) for _ in range(cfg.num_players)]
        
        # Trajectory buffer for collecting data
        self.buffer = TrajectoryBuffer()
        
        print(f"[PPO] Initialized with:")
        print(f"  clip_epsilon={cfg.clip_epsilon}, ppo_epochs={cfg.ppo_epochs}")
        print(f"  gae_lambda={cfg.gae_lambda}, gamma={cfg.gamma}")
        print(f"  value_coef={cfg.value_coef}, entropy_coef={cfg.entropy_coef}")

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

    # ----- state feature extraction -----
    
    def _extract_state_features(self, state: GameState, player_id: int) -> torch.Tensor:
        """
        Extract numerical features from game state for value network input.
        Features include: stack sizes, pot size, current bet, community cards, hole cards, etc.
        Returns: tensor of shape [feature_dim]
        """
        player = state.players[player_id]
        
        # Normalize by big blind
        bb = self.cfg.big_blind
        
        # Map betting round to numeric value
        round_map = {
            'preflop': 0.0, 'flop': 1.0, 'turn': 2.0, 'river': 3.0, 'showdown': 4.0
        }
        betting_round_val = round_map.get(state.betting_round.value, 0.0) / 4.0
        
        # Basic features
        features = [
            float(player.stack) / (10 * bb),  # Normalized stack
            float(state.pot) / (10 * bb),     # Normalized pot
            float(state.current_bet) / (10 * bb),  # Normalized current bet
            float(player.current_bet) / (10 * bb),  # Player's current bet
            1.0 if player.is_active else 0.0,  # Active flag
            betting_round_val,   # Betting round (preflop=0, flop=0.25, turn=0.5, river=0.75, showdown=1.0)
        ]
        
        # Opponent stack sizes (normalized)
        for i, p in enumerate(state.players):
            if i != player_id:
                features.append(float(p.stack) / (10 * bb))
                features.append(1.0 if p.is_active else 0.0)
        
        # Pad to fixed size (32 features)
        while len(features) < 32:
            features.append(0.0)
        features = features[:32]  # Truncate if too long
        
        return torch.tensor(features, dtype=torch.float32)

    # ----- optimizer setup -----

    def _maybe_build_optimizer(self, agents: List[AgentBase]):
        """Build optimizers for policy (LoRA) and value networks"""
        if self._policy_optim is not None:
            return
        
        # Policy optimizer (LoRA parameters)
        params = []
        for ag in agents:
            if isinstance(ag, LLMAgent):
                params += list(ag.lora_parameters())
        if params:
            self._policy_optim = AdamW(params, lr=self.cfg.learning_rate)
        else:
            self._policy_optim = None
        
        # Value optimizer
        value_params = []
        for vnet in self.value_nets:
            value_params += list(vnet.parameters())
        self._value_optim = AdamW(value_params, lr=self.cfg.learning_rate)

    # ----- GAE advantage computation -----
    
    def _compute_gae_advantages(self, rewards, values, dones):
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: list or tensor of rewards [T]
            values: list or tensor of value estimates [T]
            dones: list or tensor of done flags [T]
        
        Returns:
            advantages: tensor [T]
            returns: tensor [T] (value targets for value function)
        """
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        values = torch.tensor(values, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        
        T = len(rewards)
        advantages = torch.zeros(T, device=self.device)
        returns = torch.zeros(T, device=self.device)
        
        gae = 0.0
        next_value = 0.0  # Bootstrap value (0 for terminal state)
        
        # Compute GAE backwards from end of episode
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0.0  # Terminal state
                next_done = 1.0
            else:
                next_value = values[t + 1]
                next_done = dones[t + 1]
            
            # TD residual: r + gamma * V(s') - V(s)
            delta = rewards[t] + self.cfg.gamma * next_value * (1 - next_done) - values[t]
            
            # GAE: accumulate discounted deltas
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * (1 - next_done) * gae
            advantages[t] = gae
            
            # Return = advantage + value
            returns[t] = advantages[t] + values[t]
        
        # Normalize advantages (helps training stability)
        # Only normalize if we have more than 1 sample and std > 0
        if T > 1:
            adv_std = advantages.std()
            if adv_std > 1e-8:
                advantages = (advantages - advantages.mean()) / adv_std
        
        return advantages, returns

    # ----- PPO update -----
    
    def _ppo_update(self, agents: List[AgentBase]):
        """
        Perform PPO update using collected trajectories.
        Uses clipped surrogate objective and value function loss.
        """
        if self.buffer.size() == 0:
            print("[PPO] Warning: empty buffer, skipping update")
            return {}
        
        if self._policy_optim is None:
            print("[PPO] Warning: no policy optimizer, skipping update")
            return {}
        
        # Set models to training mode
        for vnet in self.value_nets:
            vnet.train()
        for agent in agents:
            if isinstance(agent, LLMAgent) and hasattr(agent, 'model'):
                agent.model.train()
        
        # Get trajectory data
        data = self.buffer.get_tensors(self.device)
        state_features = data['state_features']
        old_log_probs = data['log_probs']
        rewards = data['rewards']
        old_values = data['values']
        dones = data['dones']
        player_ids = data['player_ids']
        
        # Compute advantages using GAE
        advantages, returns = self._compute_gae_advantages(
            rewards.cpu().numpy(),
            old_values.cpu().numpy(),
            dones.cpu().numpy()
        )
        
        # Full PPO update: recompute current log-probs per stored prompt and run
        # multiple epochs with clipped surrogate objective.
        prompts = data.get('prompts', [])
        action_idxs = data.get('action_idxs', torch.zeros_like(player_ids))

        N = state_features.size(0)
        batch_size = min(self.cfg.batch_size, N)
        indices = np.arange(N)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(self.cfg.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, N, batch_size):
                mb_idx = indices[start:start + batch_size]
                mb_idx = list(mb_idx)

                sf_mb = state_features[mb_idx]
                old_logp_mb = old_log_probs[mb_idx]
                adv_mb = advantages[mb_idx].to(self.device)
                ret_mb = returns[mb_idx].to(self.device)
                pid_mb = player_ids[mb_idx]
                actidx_mb = action_idxs[mb_idx]

                # Recompute current log-probs and entropies per sample
                # Compute loss per sample and accumulate gradients to avoid graph reuse issues
                self._policy_optim.zero_grad()
                self._value_optim.zero_grad()
                
                total_minibatch_loss = 0.0
                for j, idx in enumerate(mb_idx):
                    pid = int(player_ids[idx].item())
                    agent = agents[pid]
                    # ensure adapter set for shared controller
                    if getattr(agent, '_controller', None) is not None:
                        agent._controller.set_adapter(agent._adapter_name)
                    prompt = prompts[idx]
                    # score_candidates_train returns (logp_vec, probs, chosen_idx, logp_chosen)
                    with torch.enable_grad():
                        logp_vec, probs, _, _ = agent.score_candidates_train(prompt, DISCRETE_ACTIONS)
                    # logp_vec is a tensor of per-candidate log-probs on agent device
                    # ensure on trainer device
                    logp_vec = logp_vec.to(self.device)
                    probs = probs.to(self.device)
                    aidx = int(actidx_mb[j].item())
                    new_logp = logp_vec[aidx]
                    old_logp = old_logp_mb[j]
                    adv = adv_mb[j]
                    ret = ret_mb[j]
                    
                    # entropy of distribution over candidates
                    ent = -(probs * torch.log(probs + 1e-12)).sum()
                    
                    # Compute per-sample policy loss
                    ratio = torch.exp(new_logp - old_logp)
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_epsilon, 1.0 + self.cfg.clip_epsilon) * adv
                    sample_policy_loss = -torch.min(surr1, surr2)
                    
                    # Compute per-sample value loss
                    v = self.value_nets[pid](sf_mb[j:j+1]).squeeze()
                    sample_value_loss = nn.functional.mse_loss(v, ret)
                    
                    # Combined loss for this sample
                    sample_loss = sample_policy_loss + self.cfg.value_coef * sample_value_loss - self.cfg.entropy_coef * ent
                    
                    # Backward and accumulate gradients
                    sample_loss.backward()
                    
                    total_minibatch_loss += sample_loss.item()
                
                # Now do the optimizer step once with accumulated gradients
                nn_utils.clip_grad_norm_(self._policy_optim.param_groups[0]['params'], self.cfg.max_grad_norm)
                nn_utils.clip_grad_norm_(self._value_optim.param_groups[0]['params'], self.cfg.max_grad_norm)
                self._policy_optim.step()
                self._value_optim.step()
                
                # For logging, compute average losses
                avg_loss = total_minibatch_loss / len(mb_idx)
                total_policy_loss += avg_loss
                total_value_loss += 0.0  # We combined them above
                total_entropy += 0.0
                n_updates += 1

        # Clear buffer after update
        self.buffer.reset()

        if n_updates == 0:
            return {}

        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
        }

    # ----- one episode with trajectory collection -----

    def play_one_episode(
        self,
        env: Optional[PokerGame] = None,
        agents: Optional[List[AgentBase]] = None
    ) -> Dict:
        """
        Play one episode and collect trajectories.
        After collecting enough data, perform PPO update.
        """
        env = env or self._init_env()
        agents = self._ensure_agents(agents)
        self._maybe_build_optimizer(agents)

        print(f"  Starting episode: init_env done", flush=True)
        init_stacks = [p.stack for p in env.players]
        state = env.reset()
        done = False
        steps = 0

        # Episode-level tracking
        episode_log_probs = {pid: [] for pid in range(self.cfg.num_players)}
        episode_rewards = {pid: [] for pid in range(self.cfg.num_players)}
        episode_values = {pid: [] for pid in range(self.cfg.num_players)}
        episode_states = {pid: [] for pid in range(self.cfg.num_players)}
        episode_prompts = {pid: [] for pid in range(self.cfg.num_players)}
        episode_action_idxs = {pid: [] for pid in range(self.cfg.num_players)}

        print(f"  Episode loop starting, max steps={self.cfg.steps_per_episode}", flush=True)
        while not done and steps < self.cfg.steps_per_episode:
            print(f"    Step {steps}: player {state.current_player().player_id} acting...", flush=True)
            current_player = state.current_player()
            pid = current_player.player_id
            legal_actions = state.get_valid_actions(current_player)

            agent = agents[pid]

            # Extract state features for value function
            state_feat = self._extract_state_features(state, pid)
            
            # Get value estimate
            with torch.no_grad():
                value_est = self.value_nets[pid](state_feat.unsqueeze(0).to(self.device))
                value_est = float(value_est.item())

            # LLM path: differentiable label selection over 10 discrete actions
            if isinstance(agent, LLMAgent) and agent.use_scoring:
                # Make sure correct adapter is active in shared-base mode
                if getattr(agent, "_controller", None) is not None:
                    agent._controller.set_adapter(agent._adapter_name)

                # Build prompt in PokerBench format
                print(f"    Building prompt for player {pid}...", flush=True)
                game_description = state.get_llm_prompt(pid)
                prompt = agent._format_prompt(game_description)
                
                # Get action probabilities and sample
                print(f"    Scoring candidates for player {pid}...", flush=True)
                logp_vec, prob_vec, idx, logp_chosen = agent.score_candidates_train(
                    prompt, DISCRETE_ACTIONS
                )
                print(f"    Scored, chosen action index: {idx}", flush=True)
                label = DISCRETE_ACTIONS[int(idx)]
                action, amount = _legalize_label(label, legal_actions, state, current_player)

                # Store trajectory data (detach log_prob to avoid graph issues in PPO update)
                episode_log_probs[pid].append(logp_chosen.detach())
                episode_states[pid].append(state_feat)
                episode_values[pid].append(value_est)
                # store prompt and chosen index so we can recompute log-probs later
                episode_prompts[pid].append(prompt)
                episode_action_idxs[pid].append(int(idx))
                # Step-level reward (0 for now, will use terminal reward)
                episode_rewards[pid].append(0.0)

                # Step env
                print(f"    Executing env.step({action}, {amount})...", flush=True)
                state, hand_complete, result = env.step(action, float(amount))
                print(f"    Step complete. hand_complete={hand_complete}, steps={steps+1}", flush=True)
                done = hand_complete
                steps += 1

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
                action=Action.CHECK,
                hand_result=result or {},
                initial_stack=init_stacks[pid],
                final_stack=final_stacks[pid],
                big_blind=self.cfg.big_blind
            )
            terminal_rewards.append(float(r))

        # Add trajectories to buffer (assign terminal reward to last step of each player)
        for pid in range(self.cfg.num_players):
            if len(episode_log_probs[pid]) > 0:
                # Assign terminal reward to last step, 0 to others
                for i, (log_prob, state_feat, value, reward, prompt, action_idx) in enumerate(zip(
                    episode_log_probs[pid],
                    episode_states[pid],
                    episode_values[pid],
                    episode_rewards[pid],
                    episode_prompts[pid],
                    episode_action_idxs[pid]
                )):
                    # Last step gets terminal reward
                    final_reward = terminal_rewards[pid] if i == len(episode_log_probs[pid]) - 1 else 0.0
                    is_done = 1.0 if i == len(episode_log_probs[pid]) - 1 else 0.0
                    
                    self.buffer.add_with_prompt(
                        state_feat=state_feat,
                        log_prob=log_prob,
                        reward=final_reward,
                        value=value,
                        done=is_done,
                        player_id=pid,
                        prompt=prompt,
                        action_idx=action_idx,
                    )

        # Perform PPO update
        update_stats = self._ppo_update(agents)

        log = {
            "steps": steps,
            "stacks_init": init_stacks,
            "stacks_final": final_stacks,
            "result": result or {},
            "terminal_rewards": terminal_rewards,
            **update_stats,
        }
        return log
