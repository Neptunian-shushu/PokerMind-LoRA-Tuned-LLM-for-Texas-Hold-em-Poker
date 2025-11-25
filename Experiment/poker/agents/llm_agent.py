from typing import List, Tuple
from .base import Agent
from .parser import PokerActionParser

class LLMAgent(Agent):
    def __init__(self, name, model_runner, parser=None):
        super().__init__(name)
        self.model_runner=model_runner
        self.parser=parser or PokerActionParser()

    def act(self, observation:str, valid_actions:List[str])->Tuple[str,float]:
        prompt=self._build_prompt(observation, valid_actions)
        llm_out=self.model_runner.generate(prompt)
        return self.parser.extract_action(llm_out, valid_actions)

    def _build_prompt(self, observation, valid_actions):
        v=", ".join(valid_actions)
        s=f"You are a specialist in playing Texas Holdem. The following will be a game scenario and you need to make the optimal decision.\nGame State:\n{observation}\nValid actions: {v}\n Decide on an action based on the strength of your hand on this board, your position, and actions before you. Do not explain your answer.\n Your optimal action is:"
        return s
