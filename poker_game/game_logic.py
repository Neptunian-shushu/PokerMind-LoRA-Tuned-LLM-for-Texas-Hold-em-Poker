"""
Core poker game logic for Texas Hold'em
Complete implementation for PPO training and demo interface
"""

from typing import List, Tuple, Optional, Dict
import random
from .deck import Deck, Card
from .game_state import GameState, PlayerState, Action, BettingRound
from .hand_evaluator import HandEvaluator


class PokerGame:
    """Texas Hold'em Poker Game Engine"""
    
    def __init__(
        self,
        num_players: int = 2,
        starting_stack: float = 100.0,
        small_blind: float = 0.5,
        big_blind: float = 1.0,
        seed: Optional[int] = None
    ):
        """
        Initialize a poker game
        
        Args:
            num_players: Number of players (2-10)
            starting_stack: Starting chip count for each player
            small_blind: Small blind amount
            big_blind: Big blind amount
            seed: Random seed for reproducibility
        """
        if num_players < 2 or num_players > 10:
            raise ValueError("Number of players must be between 2 and 10")
        
        self.num_players = num_players
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.seed = seed
        
        # Initialize players
        self.players = [
            PlayerState(
                player_id=i,
                stack=starting_stack,
                position=self._get_position_name(i, num_players)
            )
            for i in range(num_players)
        ]
        
        # Game state
        self.state = GameState(
            players=self.players,
            small_blind=small_blind,
            big_blind=big_blind,
            button_position=0
        )
        
        self.deck = Deck(seed=seed)
        self.hand_evaluator = HandEvaluator()
        
    def _get_position_name(self, idx: int, num_players: int) -> str:
        """Get position name for player"""
        if num_players == 2:
            return "BTN/SB" if idx == 0 else "BB"
        else:
            positions = ["BTN", "SB", "BB"] + [f"MP{i}" for i in range(num_players - 3)]
            return positions[idx] if idx < len(positions) else f"P{idx}"
    
    def reset(self) -> GameState:
        """Start a new hand"""
        # Move button
        self.state.hand_number += 1
        self.state.button_position = (self.state.button_position + 1) % self.num_players
        
        # Reset deck
        self.deck.reset()
        
        # Reset players for new hand
        for player in self.players:
            player.hole_cards = []
            player.current_bet = 0.0
            player.total_bet = 0.0
            player.is_active = player.stack > 0  # Only active if has chips
            player.is_all_in = False
        
        # Reset game state
        self.state.community_cards = []
        self.state.pot = 0.0
        self.state.current_bet = 0.0
        self.state.min_raise = self.big_blind
        self.state.betting_round = BettingRound.PREFLOP
        self.state.action_history = []
        
        # Post blinds
        self._post_blinds()
        
        # Deal hole cards
        self._deal_hole_cards()
        
        # Set first player to act (after big blind)
        self.state.current_player_idx = (self.state.button_position + 3) % self.num_players
        if self.num_players == 2:
            self.state.current_player_idx = self.state.button_position
        
        # Find first player who can act
        while not self.players[self.state.current_player_idx].can_act():
            self.state.next_player()
        
        return self.state
    
    def _post_blinds(self):
        """Post small and big blinds"""
        if self.num_players == 2:
            # Heads-up: button posts small blind
            sb_idx = self.state.button_position
            bb_idx = (self.state.button_position + 1) % 2
        else:
            sb_idx = (self.state.button_position + 1) % self.num_players
            bb_idx = (self.state.button_position + 2) % self.num_players
        
        # Small blind
        sb_player = self.players[sb_idx]
        sb_amount = sb_player.bet(self.small_blind)
        self.state.add_to_pot(sb_amount)
        self.state.record_action(sb_player.player_id, Action.BET, sb_amount)
        
        # Big blind
        bb_player = self.players[bb_idx]
        bb_amount = bb_player.bet(self.big_blind)
        self.state.add_to_pot(bb_amount)
        self.state.current_bet = bb_amount
        self.state.record_action(bb_player.player_id, Action.BET, bb_amount)
    
    def _deal_hole_cards(self):
        """Deal 2 cards to each active player"""
        for _ in range(2):
            for player in self.players:
                if player.is_active:
                    card = self.deck.deal_one()
                    player.hole_cards.append(card)
    
    def step(self, action: Action, amount: float = 0.0) -> Tuple[GameState, bool, Optional[Dict]]:
        """
        Execute one game action
        
        Args:
            action: The action to take
            amount: Bet/raise amount (if applicable)
        
        Returns:
            (new_state, hand_complete, result_info)
        """
        current_player = self.state.current_player()
        
        # Validate action
        valid_actions = self.state.get_valid_actions(current_player)
        if action not in valid_actions:
            raise ValueError(f"Invalid action {action} for player {current_player.player_id}. Valid: {valid_actions}")
        
        # Execute action
        self._execute_action(current_player, action, amount)
        
        # Record action
        self.state.record_action(current_player.player_id, action, amount)
        
        # Move to next player
        self.state.next_player()
        
        # Check if betting round is complete
        if self.state.is_betting_round_complete():
            return self._advance_betting_round()
        
        return self.state, False, None
    
    def _execute_action(self, player: PlayerState, action: Action, amount: float):
        """Execute a player's action"""
        if action == Action.FOLD:
            player.is_active = False
        
        elif action == Action.CHECK:
            # No chips added
            pass
        
        elif action == Action.CALL:
            call_amount = self.state.current_bet - player.current_bet
            actual_amount = player.bet(call_amount)
            self.state.add_to_pot(actual_amount)
        
        elif action == Action.BET:
            if amount < self.big_blind:
                amount = self.big_blind
            actual_amount = player.bet(amount)
            self.state.add_to_pot(actual_amount)
            self.state.current_bet = player.current_bet
            self.state.min_raise = amount
        
        elif action == Action.RAISE:
            total_bet_needed = self.state.current_bet + amount
            if amount < self.state.min_raise:
                amount = self.state.min_raise
                total_bet_needed = self.state.current_bet + amount
            
            raise_amount = total_bet_needed - player.current_bet
            actual_amount = player.bet(raise_amount)
            self.state.add_to_pot(actual_amount)
            self.state.current_bet = player.current_bet
            self.state.min_raise = amount
    
    def _advance_betting_round(self) -> Tuple[GameState, bool, Optional[Dict]]:
        """Move to next betting round or showdown"""
        active_players = self.state.get_active_players()
        
        # If only one player left, they win
        if len(active_players) == 1:
            return self._handle_winner_by_fold(active_players[0])
        
        # Reset current bets for new round
        for player in self.players:
            player.reset_current_bet()
        self.state.current_bet = 0.0
        
        # Advance to next street
        if self.state.betting_round == BettingRound.PREFLOP:
            self._deal_flop()
            self.state.betting_round = BettingRound.FLOP
        
        elif self.state.betting_round == BettingRound.FLOP:
            self._deal_turn()
            self.state.betting_round = BettingRound.TURN
        
        elif self.state.betting_round == BettingRound.TURN:
            self._deal_river()
            self.state.betting_round = BettingRound.RIVER
        
        elif self.state.betting_round == BettingRound.RIVER:
            # Go to showdown
            return self._handle_showdown()
        
        # Set first player to act (after button)
        self.state.current_player_idx = (self.state.button_position + 1) % self.num_players
        while not self.players[self.state.current_player_idx].can_act():
            self.state.next_player()
        
        # Check if betting round immediately complete (all all-in)
        if self.state.is_betting_round_complete():
            return self._advance_betting_round()
        
        return self.state, False, None
    
    def _deal_flop(self):
        """Deal the flop (3 community cards)"""
        self.deck.deal_one()  # Burn card
        self.state.community_cards.extend(self.deck.deal(3))
    
    def _deal_turn(self):
        """Deal the turn (4th community card)"""
        self.deck.deal_one()  # Burn card
        self.state.community_cards.append(self.deck.deal_one())
    
    def _deal_river(self):
        """Deal the river (5th community card)"""
        self.deck.deal_one()  # Burn card
        self.state.community_cards.append(self.deck.deal_one())
    
    def _handle_winner_by_fold(self, winner: PlayerState) -> Tuple[GameState, bool, Dict]:
        """Handle hand when all but one player folds"""
        winner.win_pot(self.state.pot)
        
        result = {
            'winners': [winner.player_id],
            'win_type': 'fold',
            'pot': self.state.pot,
            'final_board': [str(c) for c in self.state.community_cards]
        }
        
        self.state.pot = 0.0
        return self.state, True, result
    
    def _handle_showdown(self) -> Tuple[GameState, bool, Dict]:
        """Determine winner(s) at showdown"""
        self.state.betting_round = BettingRound.SHOWDOWN
        active_players = self.state.get_active_players()
        
        # Evaluate each player's hand
        player_hands = []
        for player in active_players:
            all_cards = player.hole_cards + self.state.community_cards
            hand_value = self.hand_evaluator.evaluate_hand(all_cards)
            hand_description = self.hand_evaluator.get_hand_description(all_cards)
            player_hands.append((player, hand_value, hand_description))
        
        # Sort by hand strength (best first)
        player_hands.sort(key=lambda x: (x[1][0], x[1][1]), reverse=True)
        
        # Determine winner(s) - handle ties
        winners = [player_hands[0]]
        best_hand = player_hands[0][1]
        
        for player, hand_value, hand_desc in player_hands[1:]:
            if hand_value == best_hand:
                winners.append((player, hand_value, hand_desc))
            else:
                break
        
        # Split pot among winners
        pot_share = self.state.pot / len(winners)
        winner_ids = []
        for player, hand_value, hand_desc in winners:
            player.win_pot(pot_share)
            winner_ids.append(player.player_id)
        
        result = {
            'winners': winner_ids,
            'win_type': 'showdown',
            'pot': self.state.pot,
            'pot_share': pot_share,
            'winning_hand': winners[0][2],
            'final_board': [str(c) for c in self.state.community_cards],
            'all_hands': [(p[0].player_id, p[2]) for p in player_hands]
        }
        
        self.state.pot = 0.0
        return self.state, True, result
    
    def get_game_state_string(self, player_perspective: Optional[int] = None) -> str:
        """
        Get formatted game state string for LLM
        
        Args:
            player_perspective: If provided, only show this player's hole cards
        """
        lines = []
        
        # Hand info
        lines.append(f"Hand #{self.state.hand_number} - {self.state.betting_round.value.upper()}")
        
        # Community cards
        if self.state.community_cards:
            board = " ".join(str(c) for c in self.state.community_cards)
            lines.append(f"Board: {board}")
        else:
            lines.append("Board: (no cards yet)")
        
        # Pot
        lines.append(f"Pot: ${self.state.pot:.0f}")
        
        # Players
        lines.append("\nPlayers:")
        for player in self.players:
            status = "ACTIVE" if player.is_active else "FOLDED"
            if player.is_all_in:
                status = "ALL-IN"
            
            # Show hole cards only for perspective player
            cards_str = ""
            if player_perspective is None or player.player_id == player_perspective:
                if player.hole_cards:
                    cards_str = f" [{player.hole_cards[0]} {player.hole_cards[1]}]"
            
            lines.append(f"  {player.position} (P{player.player_id}): ${player.stack:.0f} ({status}){cards_str}")
        
        # Current action
        if not self.state.is_betting_round_complete():
            current = self.state.current_player()
            lines.append(f"\nAction on: {current.position} (P{current.player_id})")
            lines.append(f"Current bet: ${self.state.current_bet:.0f}")
        
        return "\n".join(lines)
