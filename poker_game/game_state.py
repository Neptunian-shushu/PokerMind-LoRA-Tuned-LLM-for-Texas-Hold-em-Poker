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
        return f"Player{self.player_id}({self.position}): ${self.stack:.1f} [{status}]"


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
    small_blind: float = 0.5
    big_blind: float = 1.0
    hand_number: int = 0
    starting_stack: float = 100.0  # Starting chip count for all players
    last_aggressor_idx: int = -1  # Index of last player to bet/raise (-1 if none)
    num_actions_this_round: int = 0  # Count of actions taken in current betting round
    
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
        
        # If there's been an aggressor (bet/raise), action has come back to them
        # and everyone has matched, so round is complete
        if self.last_aggressor_idx >= 0:
            return True
        
        # No aggressor yet (all checks) - need all players to have acted at least once
        # In heads-up, need at least 2 actions (one from each player)
        # In multi-way, need at least num_active_players actions
        num_active = len(active_players)
        if self.num_actions_this_round >= num_active:
            return True
        
        return False
    
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
        return f"{self.betting_round.value.upper()} | Board: [{cards_str}] | Pot: ${self.pot:.1f} | {active} players"
    
    def record_action(self, player_id: int, action: Action, amount: float = 0, is_blind: bool = False):
        """Record an action in the history"""
        self.action_history.append({
            'player_id': player_id,
            'action': action.value,
            'amount': amount,
            'betting_round': self.betting_round.value,
            'pot_after': self.pot,
            'is_blind': is_blind
        })
    
    def get_llm_prompt(self, player_perspective: int) -> str:
        """
        Generate a PokerBench-style prompt for LLM decision making
        
        Args:
            player_perspective: The player ID for whom to generate the prompt
        
        Returns:
            Formatted prompt string for LLM
        """
        player = self.players[player_perspective]
        
        # Position mapping for different player counts
        position_names = [p.position for p in self.players]
        player_position = position_names[player_perspective]
        
        # Determine game format description
        num_players = len(self.players)
        if num_players == 2:
            game_format = "heads-up"
        else:
            game_format = f"{num_players}-handed"
        
        # Build the prompt
        lines = []
        
        # Header
        lines.append(f"You are a specialist in playing {game_format} No Limit Texas Holdem. The following will be a game scenario and you need to make the optimal decision.")
        lines.append("")
        lines.append("Here is a game summary:")
        lines.append("")
        
        # Game setup
        lines.append(f"The small blind is {self.small_blind} chips and the big blind is {self.big_blind} chips. Everyone started with {self.starting_stack} chips.")
        
        # Player positions
        active_positions = [position_names[i] for i, p in enumerate(self.players) if p.is_active]
        if len(active_positions) == 2:
            lines.append(f"The player positions involved in this game are SB, BB.")
        else:
            lines.append(f"The player positions involved in this game are {', '.join(active_positions)}.")
        
        # Current player info
        hole_cards_str = " and ".join(str(card) for card in player.hole_cards)
        lines.append(f"In this hand, your position is {player_position}, and your holding is [{hole_cards_str}].")
        
        # Action history narrative
        action_narrative = self._build_action_narrative(player_perspective)
        if action_narrative:
            lines.append(action_narrative)
        
        # Current board
        if self.community_cards:
            board_str = ", ".join(str(card) for card in self.community_cards[:-1])
            if len(self.community_cards) > 1:
                board_str += f", and {self.community_cards[-1]}"
            else:
                board_str = str(self.community_cards[0])
            
            if self.betting_round == BettingRound.FLOP:
                lines.append(f"The flop comes {board_str}.")
            elif self.betting_round == BettingRound.TURN:
                flop_cards = ", ".join(str(card) for card in self.community_cards[:3])
                turn_card = str(self.community_cards[3])
                lines.append(f"The flop comes {flop_cards}, the turn comes {turn_card}.")
            elif self.betting_round == BettingRound.RIVER:
                flop_cards = ", ".join(str(card) for card in self.community_cards[:3])
                turn_card = str(self.community_cards[3])
                river_card = str(self.community_cards[4])
                lines.append(f"The flop comes {flop_cards}, the turn comes {turn_card}, the river comes {river_card}.")
        
        # Current betting actions (if any in current round)
        current_round_actions = self._get_current_round_actions(player_perspective)
        if current_round_actions:
            lines.append(f" {current_round_actions}")
        
        # Current situation
        lines.append("")
        lines.append("Now it is your turn to make a move.")
        lines.append(f"To remind you, the current pot size is {self.pot} chips, and your holding is [{hole_cards_str}].")
        lines.append("")
        lines.append("Decide on an action based on the strength of your hand on this board, your position, and actions before you. Do not explain your answer.")
        lines.append("Your optimal action is:")
        
        return "\n".join(lines)
    
    def _get_position_names(self) -> List[str]:
        """Get standard poker position names for current player count"""
        num_players = len(self.players)
        
        if num_players == 2:
            return ["SB", "BB"]  # In heads-up, button is SB, other is BB
        elif num_players == 3:
            return ["BTN", "SB", "BB"]
        elif num_players == 4:
            return ["CO", "BTN", "SB", "BB"]
        elif num_players == 5:
            return ["MP", "CO", "BTN", "SB", "BB"]
        elif num_players == 6:
            return ["UTG", "HJ", "CO", "BTN", "SB", "BB"]
        elif num_players == 7:
            return ["UTG", "UTG+1", "MP", "CO", "BTN", "SB", "BB"]
        elif num_players == 8:
            return ["UTG", "UTG+1", "UTG+2", "MP", "CO", "BTN", "SB", "BB"]
        elif num_players == 9:
            return ["UTG", "UTG+1", "UTG+2", "UTG+3", "MP", "CO", "BTN", "SB", "BB"]
        else:  # 10 players
            return ["UTG", "UTG+1", "UTG+2", "UTG+3", "UTG+4", "MP", "CO", "BTN", "SB", "BB"]
    
    def _build_action_narrative(self, player_perspective: int) -> str:
        """Build narrative description of action history"""
        if not self.action_history:
            return ""
        
        position_names = [p.position for p in self.players]
        lines = []
        
        # Group actions by betting round
        rounds = {}
        for action in self.action_history:
            round_name = action['betting_round']
            if round_name not in rounds:
                rounds[round_name] = []
            rounds[round_name].append(action)
        
        # Process each round
        for round_name, actions in rounds.items():
            if round_name == 'preflop':
                preflop_actions = []
                for action in actions:
                    if not action.get('is_blind', False):  # Skip blind actions in narrative
                        player_pos = position_names[action['player_id']]
                        action_str = self._format_action(action, player_pos)
                        preflop_actions.append(action_str)
                
                if preflop_actions:
                    if len(preflop_actions) == 1:
                        lines.append(f"Before the flop, {preflop_actions[0]}.")
                    else:
                        lines.append(f"Before the flop, {', '.join(preflop_actions[:-1])}, and {preflop_actions[-1]}.")
                    lines.append("Assume that all other players that is not mentioned folded.")
        
        return " ".join(lines)
    
    def _format_action(self, action: Dict, player_pos: str) -> str:
        """Format a single action for narrative"""
        action_type = action['action']
        amount = action['amount']
        
        if action_type == 'fold':
            return f"{player_pos} fold"
        elif action_type == 'check':
            return f"{player_pos} check"
        elif action_type == 'call':
            return f"{player_pos} call"
        elif action_type == 'bet':
            return f"{player_pos} bet {amount} chips"
        elif action_type == 'raise':
            return f"{player_pos} raise {amount} chips"
        else:
            return f"{player_pos} {action_type}"
    
    def _get_current_round_actions(self, player_perspective: int) -> str:
        """Get actions in the current betting round so far"""
        # For preflop, actions are already included in the main narrative
        if self.betting_round.value == 'preflop':
            return ""
            
        position_names = [p.position for p in self.players]
        current_actions = []
        
        for action in self.action_history:
            if action['betting_round'] == self.betting_round.value and not action.get('is_blind', False):
                player_pos = position_names[action['player_id']]
                action_str = self._format_action(action, player_pos)
                current_actions.append(action_str)
        
        if current_actions:
            if len(current_actions) == 1:
                return f"then {current_actions[0]}."
            else:
                return f"then {', '.join(current_actions[:-1])}, and {current_actions[-1]}."
        
        return ""
    
    def __repr__(self) -> str:
        return self.get_state_string()
