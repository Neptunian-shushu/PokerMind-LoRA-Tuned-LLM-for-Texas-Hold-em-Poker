"""
PPO Hyperparameters and Configuration
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PPOConfig:
    """Configuration for PPO training"""
    
    # Model paths
    base_model_path: str = "/home/hice1/bshu30/CS6220/poker-lora-model/Meta-Llama-3-8B/"
    output_dir: str = "/home/hice1/bshu30/CS6220/ppo/ppo_checkpoints/"
    log_dir: str = "/home/hice1/bshu30/CS6220/logs/ppo/"
    
    # Training hyperparameters
    num_episodes: int = 10000  # Number of self-play games
    steps_per_episode: int = 50  # Max steps per game (prevent infinite loops)
    batch_size: int = 32  # Batch size for PPO updates
    epochs_per_update: int = 4  # PPO epochs per batch
    learning_rate: float = 1e-6  # Learning rate for policy updates
    
    # PPO algorithm parameters
    gamma: float = 0.99  # Discount factor for rewards
    gae_lambda: float = 0.95  # GAE lambda for advantage estimation
    clip_epsilon: float = 0.2  # PPO clipping parameter
    value_loss_coef: float = 0.5  # Coefficient for value loss
    entropy_coef: float = 0.01  # Entropy bonus for exploration
    max_grad_norm: float = 0.5  # Gradient clipping
    
    # KL divergence constraint (keep policy close to reference)
    kl_coef: float = 0.1  # Coefficient for KL penalty
    target_kl: float = 0.01  # Target KL divergence
    adaptive_kl: bool = True  # Adapt KL coefficient during training
    
    # Poker game settings
    num_players: int = 2  # Number of players in self-play
    starting_stack: float = 100.0  # Starting chips
    small_blind: float = 0.5  # Small blind amount
    big_blind: float = 1.0  # Big blind amount
    
    # Reward shaping
    win_reward: float = 1.0  # Reward for winning a hand
    lose_penalty: float = -1.0  # Penalty for losing
    fold_penalty: float = -0.1  # Small penalty for folding (encourage playing)
    showdown_bonus: float = 0.2  # Bonus for reaching showdown
    
    # Training efficiency
    num_workers: int = 1  # Parallel game environments (future enhancement)
    gradient_accumulation_steps: int = 4  # Accumulate gradients
    fp16: bool = True  # Use mixed precision training
    
    # Evaluation
    eval_frequency: int = 500  # Evaluate every N episodes
    eval_episodes: int = 100  # Number of games for evaluation
    save_frequency: int = 1000  # Save checkpoint every N episodes
    
    # Model settings
    max_sequence_length: int = 512  # Max tokens for LLM input
    temperature: float = 0.8  # Sampling temperature
    top_p: float = 0.9  # Nucleus sampling
    
    # Logging
    log_frequency: int = 10  # Log metrics every N episodes
    wandb_project: Optional[str] = None  # W&B project name (optional)
    wandb_entity: Optional[str] = None  # W&B entity
    
    # Reproducibility
    seed: int = 42
    
    def __post_init__(self):
        """Validate configuration"""
        assert 0 < self.gamma <= 1, "gamma must be in (0, 1]"
        assert 0 < self.gae_lambda <= 1, "gae_lambda must be in (0, 1]"
        assert self.clip_epsilon > 0, "clip_epsilon must be positive"
        assert self.kl_coef >= 0, "kl_coef must be non-negative"
        assert self.num_players >= 2, "num_players must be at least 2"
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


# Default configuration
DEFAULT_CONFIG = PPOConfig()


# Fast training config (for testing)
FAST_CONFIG = PPOConfig(
    num_episodes=100,
    batch_size=16,
    eval_frequency=50,
    save_frequency=50
)


# Production config (full training)
PRODUCTION_CONFIG = PPOConfig(
    num_episodes=50000,
    batch_size=64,
    epochs_per_update=4,
    learning_rate=5e-7,
    eval_frequency=1000,
    save_frequency=2000,
    gradient_accumulation_steps=8
)
