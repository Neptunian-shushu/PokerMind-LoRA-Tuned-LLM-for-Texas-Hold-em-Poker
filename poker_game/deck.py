"""
Deck and Card classes for Texas Hold'em Poker
"""

import random
from typing import List, Optional
from enum import Enum


class Suit(Enum):
    """Card suits"""
    HEARTS = 'Heart'
    DIAMONDS = 'Diamond'
    CLUBS = 'Club'
    SPADES = 'Spade'


class Rank(Enum):
    """Card ranks with values for comparison"""
    TWO = (2, 'Two')
    THREE = (3, 'Three')
    FOUR = (4, 'Four')
    FIVE = (5, 'Five')
    SIX = (6, 'Six')
    SEVEN = (7, 'Seven')
    EIGHT = (8, 'Eight')
    NINE = (9, 'Nine')
    TEN = (10, 'Ten')
    JACK = (11, 'Jack')
    QUEEN = (12, 'Queen')
    KING = (13, 'King')
    ACE = (14, 'Ace')
    
    def __init__(self, value, symbol):
        self.value = value
        self.symbol = symbol


class Card:
    """Represents a single playing card"""
    
    def __init__(self, rank: Rank, suit: Suit):
        self.rank = rank
        self.suit = suit
    
    def __str__(self) -> str:
        return f"{self.rank.symbol} Of {self.suit.value}"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Card):
            return False
        return self.rank == other.rank and self.suit == other.suit
    
    def __hash__(self) -> int:
        return hash((self.rank, self.suit))
    
    @classmethod
    def from_string(cls, card_str: str) -> 'Card':
        """Create card from string like 'As' or 'Kh' or 'Ace Of Heart'"""
        rank_map = {
            '2': Rank.TWO, 'two': Rank.TWO,
            '3': Rank.THREE, 'three': Rank.THREE,
            '4': Rank.FOUR, 'four': Rank.FOUR,
            '5': Rank.FIVE, 'five': Rank.FIVE,
            '6': Rank.SIX, 'six': Rank.SIX,
            '7': Rank.SEVEN, 'seven': Rank.SEVEN,
            '8': Rank.EIGHT, 'eight': Rank.EIGHT,
            '9': Rank.NINE, 'nine': Rank.NINE,
            't': Rank.TEN, 'T': Rank.TEN, '10': Rank.TEN, 'ten': Rank.TEN,
            'j': Rank.JACK, 'J': Rank.JACK, 'jack': Rank.JACK,
            'q': Rank.QUEEN, 'Q': Rank.QUEEN, 'queen': Rank.QUEEN,
            'k': Rank.KING, 'K': Rank.KING, 'king': Rank.KING,
            'a': Rank.ACE, 'A': Rank.ACE, 'ace': Rank.ACE
        }
        suit_map = {
            'h': Suit.HEARTS, 'heart': Suit.HEARTS,
            'd': Suit.DIAMONDS, 'diamond': Suit.DIAMONDS,
            'c': Suit.CLUBS, 'club': Suit.CLUBS,
            's': Suit.SPADES, 'spade': Suit.SPADES
        }
        
        # Handle "Ace Of Heart" format
        if ' of ' in card_str.lower():
            parts = card_str.lower().split(' of ')
            if len(parts) == 2:
                rank_str = parts[0].strip()
                suit_str = parts[1].strip()
                
                if rank_str not in rank_map:
                    raise ValueError(f"Invalid rank: {rank_str}")
                if suit_str not in suit_map:
                    raise ValueError(f"Invalid suit: {suit_str}")
                
                return cls(rank_map[rank_str], suit_map[suit_str])
        
        # Handle "As" or "Kh" format
        if len(card_str) < 2:
            raise ValueError(f"Invalid card string: {card_str}")
        
        rank_str = card_str[0].upper()
        suit_str = card_str[1].lower()
        
        if rank_str not in rank_map:
            raise ValueError(f"Invalid rank: {rank_str}")
        if suit_str not in suit_map:
            raise ValueError(f"Invalid suit: {suit_str}")
        
        return cls(rank_map[rank_str], suit_map[suit_str])


class Deck:
    """52-card deck for Texas Hold'em"""
    
    def __init__(self, seed: Optional[int] = None):
        self.cards: List[Card] = []
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        self.reset()
    
    def reset(self):
        """Create a fresh shuffled deck"""
        self.cards = [
            Card(rank, suit)
            for suit in Suit
            for rank in Rank
        ]
        self.shuffle()
    
    def shuffle(self):
        """Shuffle the deck"""
        random.shuffle(self.cards)
    
    def deal(self, n: int = 1) -> List[Card]:
        """Deal n cards from the top of the deck"""
        if n > len(self.cards):
            raise ValueError(f"Not enough cards in deck. Requested {n}, have {len(self.cards)}")
        
        dealt_cards = self.cards[:n]
        self.cards = self.cards[n:]
        return dealt_cards
    
    def deal_one(self) -> Card:
        """Deal a single card"""
        return self.deal(1)[0]
    
    def cards_remaining(self) -> int:
        """Number of cards left in deck"""
        return len(self.cards)
    
    def __len__(self) -> int:
        return len(self.cards)
    
    def __str__(self) -> str:
        return f"Deck({len(self.cards)} cards remaining)"
