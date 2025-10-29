# ppo/config.py
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class PPOConfig:
    # Model & adapters
    base_repo_or_path: str = "meta-llama/Meta-Llama-3-8B"
    adapter_paths: List[str] = field(default_factory=lambda: [
        "/home/hice1/yli3776/PokerMind-LoRA-Tuned-LLM-for-Texas-Hold-em-Poker/poker-lora-model/Meta-Llama-3-8B"
    ])
    adapter_register_names: List[str] = field(default_factory=lambda: ["A"])
    seat_adapter_names: List[str] = field(default_factory=lambda: ["A", "A"])

    # IO
    output_dir: str = "/home/hice1/yli3776/PokerMind-LoRA-Tuned-LLM-for-Texas-Hold-em-Poker/ppo/ppo_checkpoints/"
    log_dir: str    = "/home/hice1/yli3776/PokerMind-LoRA-Tuned-LLM-for-Texas-Hold-em-Poker/logs/ppo/"
    rl_adapter_save_dir: str = "/home/hice1/yli3776/PokerMind-LoRA-Tuned-LLM-for-Texas-Hold-em-Poker/ppo/rl_adapters/"

    # Training loop
    num_episodes: int = 10000
    steps_per_episode: int = 50
    learning_rate: float = 1e-6
    log_frequency: int = 10
    save_frequency: int = 1000
    save_adapter_every: int = 1000

    # Eval
    eval_frequency: int = 500
    eval_episodes: int = 100

    # Poker env
    num_players: int = 2
    starting_stack: float = 100.0
    small_blind: float = 0.5
    big_blind: float = 1.0

    # LLM inference
    max_seq_len: int = 512
    temperature: float = 0.8
    # top_p: float = 0.9
    use_scoring: bool = True
    device_map: str = "auto"
    torch_dtype: str = "float16"

    # Repro
    seed: int = 42
    
    # RL training
    max_grad_norm: float = 1.0

    def __post_init__(self):
        assert self.num_players >= 2
        assert self.big_blind > 0
        if len(self.seat_adapter_names) != self.num_players:
            if len(self.seat_adapter_names) == 1:
                self.seat_adapter_names = [self.seat_adapter_names[0] for _ in range(self.num_players)]
            else:
                raise ValueError("seat_adapter_names must have length == num_players")

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


# Create default configuration instances
DEFAULT_CONFIG = PPOConfig()

# Fast config for testing (smaller episodes, more frequent logging)
FAST_CONFIG = PPOConfig(
    num_episodes=100,
    steps_per_episode=20,
    log_frequency=5,
    save_frequency=50,
    eval_frequency=25,
    eval_episodes=10
)