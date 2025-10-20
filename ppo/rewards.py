"""
Reward calculation functions for PPO poker training
"""

from typing import Dict, List
import numpy as np
from poker_game import GameState, PlayerState, Action


class RewardCalculator:
    """Calculate rewards for poker actions and outcomes"""
    
    def __init__(
        self,
        win_reward: float = 1.0,
        lose_penalty: float = -1.0,
        fold_penalty: float = -0.1,
        showdown_bonus: float = 0.2
    ):
        self.win_reward = win_reward
        self.lose_penalty = lose_penalty
        self.fold_penalty = fold_penalty
        self.showdown_bonus = showdown_bonus
    
    def calculate_reward(
        self,
        player_id: int,
        action: Action,
        hand_result: Dict,
        initial_stack: float,
        final_stack: float
    ) -> float:
        """
        Calculate reward for a player's performance in a hand
        
        Args:
            player_id: ID of the player
            action: Last action taken
            hand_result: Result dictionary from game completion
            initial_stack: Stack before hand
            final_stack: Stack after hand
        
        Returns:
            Reward value (float)
        """
        # Base reward: change in stack size (normalized)
        stack_change = final_stack - initial_stack
        base_reward = stack_change / initial_stack  # Normalize by starting stack
        
        # Win/loss bonus
        if player_id in hand_result.get('winners', []):
            reward = base_reward + self.win_reward
            
            # Bonus for winning at showdown (better play)
            if hand_result.get('win_type') == 'showdown':
                reward += self.showdown_bonus
        else:
            reward = base_reward + self.lose_penalty
            
            # Penalty for folding (encourage aggressive play)
            if action == Action.FOLD:
                reward += self.fold_penalty
        
        return reward
    
    def calculate_step_reward(
        self,
        state: GameState,
        action: Action,
        player_id: int
    ) -> float:
        """
        Calculate immediate step reward (sparse - mostly 0)
        Main rewards come at hand completion
        
        Args:
            state: Current game state
            action: Action taken
            player_id: Player who acted
        
        Returns:
            Step reward (usually 0, small penalties for bad play)
        """
        # Most rewards are at hand completion, but we can add small shaping
        reward = 0.0
        
        # Small penalty for folding with strong hands (future enhancement)
        # This would require hand strength evaluation
        
        # Small reward for staying in pot (encourages play)
        if action in [Action.CALL, Action.CHECK, Action.BET, Action.RAISE]:
            reward += 0.01
        
        return reward
    
    def calculate_batch_rewards(
        self,
        experiences: List[Dict],
        gamma: float = 0.99
    ) -> np.ndarray:
        """
        Calculate discounted returns for a batch of experiences
        
        Args:
            experiences: List of experience dictionaries
            gamma: Discount factor
        
        Returns:
            Array of discounted returns
        """
        rewards = [exp['reward'] for exp in experiences]
        returns = []
        
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        
        return np.array(returns)
    
    def calculate_advantages(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ) -> np.ndarray:
        """
        Calculate GAE (Generalized Advantage Estimation)
        
        Args:
            rewards: Array of rewards
            values: Array of value estimates
            dones: Array of done flags
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        
        Returns:
            Array of advantages
        """
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        last_value = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            mask = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * mask - values[t]
            advantages[t] = last_advantage = delta + gamma * gae_lambda * mask * last_advantage
        
        return advantages


class RewardShaper:
    """Advanced reward shaping for better learning"""
    
    @staticmethod
    def normalize_rewards(rewards: np.ndarray) -> np.ndarray:
        """Normalize rewards to have mean 0 and std 1"""
        if len(rewards) == 0:
            return rewards
        mean = np.mean(rewards)
        std = np.std(rewards)
        if std < 1e-8:
            return rewards - mean
        return (rewards - mean) / (std + 1e-8)
    
    @staticmethod
    def clip_rewards(rewards: np.ndarray, min_val: float = -10.0, max_val: float = 10.0) -> np.ndarray:
        """Clip rewards to prevent extreme values"""
        return np.clip(rewards, min_val, max_val)
    
    @staticmethod
    def add_exploration_bonus(
        rewards: np.ndarray,
        action_probs: np.ndarray,
        bonus_coef: float = 0.01
    ) -> np.ndarray:
        """Add entropy bonus for exploration"""
        # Higher bonus for more diverse actions
        entropy = -np.sum(action_probs * np.log(action_probs + 1e-10), axis=-1)
        return rewards + bonus_coef * entropy


# Example usage
def test_reward_calculator():
    """Test reward calculation"""
    calc = RewardCalculator()
    
    # Player wins at showdown
    hand_result = {
        'winners': [0],
        'win_type': 'showdown',
        'pot': 20.0
    }
    
    reward = calc.calculate_reward(
        player_id=0,
        action=Action.CALL,
        hand_result=hand_result,
        initial_stack=200.0,
        final_stack=210.0
    )
    
    print(f"Showdown win reward: {reward:.4f}")
    
    # Player loses by folding
    hand_result = {
        'winners': [1],
        'win_type': 'fold',
        'pot': 10.0
    }
    
    reward = calc.calculate_reward(
        player_id=0,
        action=Action.FOLD,
        hand_result=hand_result,
        initial_stack=200.0,
        final_stack=198.0
    )
    
    print(f"Fold loss reward: {reward:.4f}")


if __name__ == "__main__":
    test_reward_calculator()
