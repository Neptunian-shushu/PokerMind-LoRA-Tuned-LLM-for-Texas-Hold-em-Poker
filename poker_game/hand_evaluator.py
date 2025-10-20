"""
Hand evaluator for Texas Hold'em poker
Determines hand rankings and winner at showdown
"""

from typing import List, Tuple
from collections import Counter
from .deck import Card, Rank


class HandRank:
    """Hand ranking constants"""
    HIGH_CARD = 1
    PAIR = 2
    TWO_PAIR = 3
    THREE_OF_A_KIND = 4
    STRAIGHT = 5
    FLUSH = 6
    FULL_HOUSE = 7
    FOUR_OF_A_KIND = 8
    STRAIGHT_FLUSH = 9
    
    NAMES = {
        1: "High Card",
        2: "Pair",
        3: "Two Pair",
        4: "Three of a Kind",
        5: "Straight",
        6: "Flush",
        7: "Full House",
        8: "Four of a Kind",
        9: "Straight Flush"
    }


class HandEvaluator:
    """Evaluates poker hands and determines winners"""
    
    @staticmethod
    def evaluate_hand(cards: List[Card]) -> Tuple[int, List[int]]:
        """
        Evaluate a 5-7 card poker hand
        
        Args:
            cards: List of 5-7 cards
        
        Returns:
            (hand_rank, tiebreakers) where tiebreakers are rank values for comparison
        """
        if len(cards) < 5:
            raise ValueError(f"Need at least 5 cards to evaluate, got {len(cards)}")
        
        # If more than 5 cards, find best 5-card combination
        if len(cards) > 5:
            best_hand = HandEvaluator._find_best_five_cards(cards)
            return HandEvaluator.evaluate_hand(best_hand)
        
        # Evaluate the 5-card hand
        ranks = sorted([c.rank.value for c in cards], reverse=True)
        suits = [c.suit for c in cards]
        rank_counts = Counter(ranks)
        
        is_flush = len(set(suits)) == 1
        is_straight, straight_high = HandEvaluator._check_straight(ranks)
        
        # Straight Flush
        if is_flush and is_straight:
            return (HandRank.STRAIGHT_FLUSH, [straight_high])
        
        # Four of a Kind
        if 4 in rank_counts.values():
            quad_rank = [r for r, c in rank_counts.items() if c == 4][0]
            kicker = [r for r in ranks if r != quad_rank][0]
            return (HandRank.FOUR_OF_A_KIND, [quad_rank, kicker])
        
        # Full House
        if 3 in rank_counts.values() and 2 in rank_counts.values():
            trip_rank = [r for r, c in rank_counts.items() if c == 3][0]
            pair_rank = [r for r, c in rank_counts.items() if c == 2][0]
            return (HandRank.FULL_HOUSE, [trip_rank, pair_rank])
        
        # Flush
        if is_flush:
            return (HandRank.FLUSH, ranks)
        
        # Straight
        if is_straight:
            return (HandRank.STRAIGHT, [straight_high])
        
        # Three of a Kind
        if 3 in rank_counts.values():
            trip_rank = [r for r, c in rank_counts.items() if c == 3][0]
            kickers = sorted([r for r in ranks if r != trip_rank], reverse=True)
            return (HandRank.THREE_OF_A_KIND, [trip_rank] + kickers)
        
        # Two Pair
        pairs = [r for r, c in rank_counts.items() if c == 2]
        if len(pairs) == 2:
            pairs.sort(reverse=True)
            kicker = [r for r in ranks if r not in pairs][0]
            return (HandRank.TWO_PAIR, pairs + [kicker])
        
        # One Pair
        if len(pairs) == 1:
            pair_rank = pairs[0]
            kickers = sorted([r for r in ranks if r != pair_rank], reverse=True)
            return (HandRank.PAIR, [pair_rank] + kickers)
        
        # High Card
        return (HandRank.HIGH_CARD, ranks)
    
    @staticmethod
    def _check_straight(ranks: List[int]) -> Tuple[bool, int]:
        """Check if ranks form a straight, return (is_straight, high_card)"""
        # Check regular straight
        if ranks[0] - ranks[4] == 4 and len(set(ranks)) == 5:
            return (True, ranks[0])
        
        # Check wheel (A-2-3-4-5)
        if set(ranks) == {14, 5, 4, 3, 2}:
            return (True, 5)  # In wheel, 5 is high card
        
        return (False, 0)
    
    @staticmethod
    def _find_best_five_cards(cards: List[Card]) -> List[Card]:
        """Find best 5-card combination from 6 or 7 cards"""
        from itertools import combinations
        
        best_hand = None
        best_score = (0, [])
        
        for combo in combinations(cards, 5):
            score = HandEvaluator.evaluate_hand(list(combo))
            if HandEvaluator._compare_hands(score, best_score) > 0:
                best_score = score
                best_hand = list(combo)
        
        return best_hand
    
    @staticmethod
    def _compare_hands(hand1: Tuple[int, List[int]], hand2: Tuple[int, List[int]]) -> int:
        """
        Compare two hands
        Returns: 1 if hand1 wins, -1 if hand2 wins, 0 if tie
        """
        rank1, tiebreakers1 = hand1
        rank2, tiebreakers2 = hand2
        
        # Compare hand ranks first
        if rank1 > rank2:
            return 1
        if rank1 < rank2:
            return -1
        
        # Same hand rank, compare tiebreakers
        for t1, t2 in zip(tiebreakers1, tiebreakers2):
            if t1 > t2:
                return 1
            if t1 < t2:
                return -1
        
        return 0  # Complete tie
    
    @staticmethod
    def compare_hands(cards1: List[Card], cards2: List[Card]) -> int:
        """
        Compare two hands (public API)
        Returns: 1 if cards1 wins, -1 if cards2 wins, 0 if tie
        """
        hand1 = HandEvaluator.evaluate_hand(cards1)
        hand2 = HandEvaluator.evaluate_hand(cards2)
        return HandEvaluator._compare_hands(hand1, hand2)
    
    @staticmethod
    def get_hand_description(cards: List[Card]) -> str:
        """Get human-readable description of hand"""
        hand_rank, tiebreakers = HandEvaluator.evaluate_hand(cards)
        return HandRank.NAMES[hand_rank]
