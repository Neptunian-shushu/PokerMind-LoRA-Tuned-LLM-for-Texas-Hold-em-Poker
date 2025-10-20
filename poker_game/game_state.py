"""
Game state representation for Texas Hold'em poker
"""

from typing import List, Optional, Dict
from dataclasses import dataclass, field
from enum import Enum
from .deck import Card


class Action(Enum):
    """Possible player actions"""
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"


class BettingRound(Enum):
    """Stages of a poker hand"""
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"


@dataclass
class PlayerState:
    """State of a single player"""
    player_id: int
    stack: float  # Chip count
    hole_cards: List[Card] = field(default_factory=list)
    current_bet: float = 0.0  # Amount bet in current round
    total_bet: float = 0.0  # Total bet in this hand
    is_active: bool = True  # Still in the hand (not folded)
    is_all_in: bool = False
    position: str = ""  # "button", "sb", "bb", etc.
    
    def can_act(self) -> bool:
        """Check if player can still make actions"""
        return self.is_active and not self.is_all_in and self.stack > 0
    
    def bet(self, amount: float) -> float:
        """Place a bet, returns actual amount bet (handles all-in)"""
        actual_amount = min(amount, self.stack)
        self.stack -= actual_amount
        self.current_bet += actual_amount
        self.total_bet += actual_amount
        
        if self.stack == 0:
            self.is_all_in = True
        
        return actual_amount
    
    def reset_current_bet(self):
        """Reset current bet for new betting round"""
        self.current_bet = 0.0
    
    def win_pot(self, amount: float):
        """Award pot to player"""
        self.stack += amount
    
    def __repr__(self) -> str:
        status = "ACTIVE" if self.is_active else "FOLDED"
        if self.is_all_in:
            status = "ALL-IN"
        return f"Player{self.player_id}({self.position}): ${self.stack:.0f} [{status}]"


@dataclass
class GameState:
    """Complete state of a poker game"""
    players: List[PlayerState]
    community_cards: List[Card] = field(default_factory=list)
    pot: float = 0.0
    current_bet: float = 0.0  # Current bet to match
    min_raise: float = 0.0  # Minimum raise amount
    betting_round: BettingRound = BettingRound.PREFLOP
    button_position: int = 0  # Dealer button index
    current_player_idx: int = 0  # Whose turn it is
    small_blind: float = 1.0
    big_blind: float = 2.0
    hand_number: int = 0
    
    # Action history for this hand
    action_history: List[Dict] = field(default_factory=list)
    
    def get_active_players(self) -> List[PlayerState]:
        """Get players still in the hand"""
        return [p for p in self.players if p.is_active]
    
    def get_players_who_can_act(self) -> List[PlayerState]:
        """Get players who can still make decisions"""
        return [p for p in self.players if p.can_act()]
    
    def current_player(self) -> PlayerState:
        """Get the player whose turn it is"""
        return self.players[self.current_player_idx]
    
    def next_player(self):
        """Move to next active player"""
        start_idx = self.current_player_idx
        while True:
            self.current_player_idx = (self.current_player_idx + 1) % len(self.players)
            if self.players[self.current_player_idx].can_act():
                break
            # Prevent infinite loop if no one can act
            if self.current_player_idx == start_idx:
                break
    
    def add_to_pot(self, amount: float):
        """Add chips to the pot"""
        self.pot += amount
    
    def is_betting_round_complete(self) -> bool:
        """Check if current betting round is done"""
        active_players = self.get_active_players()
        
        if len(active_players) <= 1:
            return True  # Only one player left
        
        players_can_act = self.get_players_who_can_act()
        
        if len(players_can_act) == 0:
            return True  # All players all-in or have acted
        
        # Check if all active players have matched the current bet
        for player in active_players:
            if player.can_act() and player.current_bet < self.current_bet:
                return False
        
        return True
    
    def get_valid_actions(self, player: PlayerState) -> List[Action]:
        """Get valid actions for a player"""
        if not player.can_act():
            return []
        
        actions = []
        
        # Can always fold
        actions.append(Action.FOLD)
        
        # Check or call
        if player.current_bet == self.current_bet:
            actions.append(Action.CHECK)
        else:
            call_amount = self.current_bet - player.current_bet
            if call_amount <= player.stack:
                actions.append(Action.CALL)
            else:
                # If can't afford to call, CALL will automatically be all-in
                actions.append(Action.CALL)
                return actions
        
        # Bet or raise
        min_bet_amount = self.current_bet + self.min_raise
        if player.stack > min_bet_amount - player.current_bet:
            if self.current_bet == 0:
                actions.append(Action.BET)
            else:
                actions.append(Action.RAISE)
        
        return actions
    
    def get_state_string(self) -> str:
        """Get human-readable state description"""
        cards_str = " ".join(str(c) for c in self.community_cards)
        active = len(self.get_active_players())
        return f"{self.betting_round.value.upper()} | Board: [{cards_str}] | Pot: ${self.pot:.0f} | {active} players"
    
    def record_action(self, player_id: int, action: Action, amount: float = 0):
        """Record an action in the history"""
        self.action_history.append({
            'player_id': player_id,
            'action': action.value,
            'amount': amount,
            'betting_round': self.betting_round.value,
            'pot_after': self.pot
        })
    
    def __repr__(self) -> str:
        return self.get_state_string()
