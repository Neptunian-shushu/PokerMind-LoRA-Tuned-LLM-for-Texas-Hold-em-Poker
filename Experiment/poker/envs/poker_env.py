from typing import List
from ..game_logic import PokerGame
from ..game_state import Action

class PokerEnv:
    def __init__(self, num_players=2, starting_stack=100.0, small_blind=0.5, big_blind=1.0, seed=None):
        self.game = PokerGame(num_players, starting_stack, small_blind, big_blind, seed)
        self.num_players=num_players

    def reset(self):
        return self.game.reset()

    def step(self, action:Action, amount:float=0.0):
        return self.game.step(action, amount)

    def get_observation(self, pid:int)->str:
        return self.game.get_game_state_string(player_perspective=pid)

    def get_valid_actions_for_player(self, pid:int)->List[str]:
        st=self.game.state
        player=self.game.players[pid]
        acts=st.get_valid_actions(player)
        m={Action.FOLD:"fold",Action.CHECK:"check",Action.CALL:"call",Action.BET:"bet",Action.RAISE:"raise"}
        return [m[a] for a in acts]

    def string_to_action_enum(self,name:str):
        name=name.lower()
        return {"fold":Action.FOLD,"check":Action.CHECK,"call":Action.CALL,"bet":Action.BET,"raise":Action.RAISE}[name]
