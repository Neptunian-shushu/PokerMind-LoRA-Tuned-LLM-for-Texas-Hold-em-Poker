# ppo/rewards.py
from dataclasses import dataclass

@dataclass
class RewardCalculator:
    big_blind: float = 1.0

    def __post_init__(self):
        assert self.big_blind > 0, "big_blind must be positive"
        self.big_blind = float(self.big_blind)

    def calculate_reward(self, player_id, action, hand_result, initial_stack, final_stack, big_blind=None) -> float:
        bb = float(big_blind) if (big_blind is not None and big_blind > 0) else self.big_blind
        return (float(final_stack) - float(initial_stack)) / bb