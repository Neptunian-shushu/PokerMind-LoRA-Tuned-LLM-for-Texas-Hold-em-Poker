import re
from typing import Tuple, List

class PokerActionParser:
    def __init__(self, default_action: str="fold"):
        self.default_action = default_action

    def extract_action(self, text: str, valid_actions: List[str]) -> Tuple[str,float]:
        if not text:
            return self._fallback(valid_actions)
        t = text.strip().lower()
        first = t.splitlines()[0].strip()
        tokens = first.split()
        if not tokens:
            return self._fallback(valid_actions)
        raw = tokens[0]
        action=None
        for a in valid_actions:
            if raw.startswith(a):
                action=a; break
        if action is None:
            for a in ["fold","check","call","bet","raise"]:
                if a in valid_actions and a in first:
                    action=a; break
        if action is None:
            return self._fallback(valid_actions)
        amount=0.0
        if action in ("bet","raise"):
            m=re.search(r"(\d+(\.\d+)?)", first)
            if m:
                amount=float(m.group(1))
        return action, amount

    def _fallback(self, valid_actions):
        if self.default_action in valid_actions:
            return self.default_action,0.0
        if valid_actions:
            return valid_actions[0],0.0
        return "fold",0.0
