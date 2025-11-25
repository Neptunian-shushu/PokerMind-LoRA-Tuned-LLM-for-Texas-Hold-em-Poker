from typing import Tuple, List

class Agent:
    def __init__(self, name: str):
        self.name = name

    def act(self, observation: str, valid_actions: List[str]) -> Tuple[str, float]:
        raise NotImplementedError
