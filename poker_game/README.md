# Poker Game Module

Reusable Texas Hold'em poker game engine for PPO training and demo interface.

## Components

### `deck.py` - Card and Deck Management
- **Card**: Represents individual playing cards with rank and suit
- **Deck**: 52-card deck with shuffling and dealing

**Usage:**
```python
from poker_game import Deck, Card

deck = Deck(seed=42)  # Reproducible shuffling
cards = deck.deal(5)  # Deal 5 cards
print(cards)  # [Ah, Kd, Qs, Jc, Ts]
```

### `hand_evaluator.py` - Hand Ranking
- Evaluates 5-7 card poker hands
- Determines winners at showdown
- Provides hand strength estimates

**Hand Rankings:**
1. High Card
2. Pair
3. Two Pair
4. Three of a Kind
5. Straight
6. Flush
7. Full House
8. Four of a Kind
9. Straight Flush

**Usage:**
```python
from poker_game import HandEvaluator, Card

cards = [Card.from_string(s) for s in ['Ah', 'Ad', 'Kh', 'Kd', 'Qs']]
hand_rank, tiebreakers = HandEvaluator.evaluate_hand(cards)
description = HandEvaluator.get_hand_description(cards)
print(description)  # "Two Pair"
```

### `game_state.py` - State Representation
- **PlayerState**: Individual player information (stack, cards, bets)
- **GameState**: Complete game state (pot, community cards, betting round)
- **Action**: Enum of possible actions (FOLD, CHECK, CALL, BET, RAISE)
- **BettingRound**: Enum for game stages (PREFLOP, FLOP, TURN, RIVER, SHOWDOWN)

**Usage:**
```python
from poker_game import GameState, PlayerState, Action

player = PlayerState(player_id=0, stack=200.0)
player.bet(10.0)  # Place a bet
print(player.stack)  # 190.0
```

### `game_logic.py` - Core Game Engine
- **PokerGame**: Complete Texas Hold'em implementation
- Handles blinds, dealing, betting rounds, showdown
- Supports 2-10 players
- Action validation and execution

**Usage:**
```python
from poker_game import PokerGame, Action

# Create a game
game = PokerGame(num_players=2, starting_stack=200.0, small_blind=1.0, big_blind=2.0)

# Start a hand
state = game.reset()

# Get valid actions for current player
player = state.current_player()
valid_actions = state.get_valid_actions(player)

# Execute an action
new_state, hand_complete, result = game.step(Action.RAISE, amount=6.0)

# Print game state
print(game.get_game_state_string())
```

## Complete Game Flow Example

```python
from poker_game import PokerGame, Action

# Initialize game
game = PokerGame(num_players=2, starting_stack=200.0)

# Play a hand
state = game.reset()
print(f"Starting hand #{state.hand_number}")

while True:
    # Get current player
    current_player = state.current_player()
    print(f"\n{game.get_game_state_string(player_perspective=current_player.player_id)}")
    
    # Get valid actions
    valid_actions = state.get_valid_actions(current_player)
    print(f"Valid actions: {[a.value for a in valid_actions]}")
    
    # Choose action (simplified - real implementation would use AI model)
    if Action.CHECK in valid_actions:
        action = Action.CHECK
    elif Action.CALL in valid_actions:
        action = Action.CALL
    else:
        action = Action.FOLD
    
    # Execute action
    state, hand_complete, result = game.step(action)
    
    # Check if hand is over
    if hand_complete:
        print(f"\n=== HAND COMPLETE ===")
        print(f"Winners: Player {result['winners']}")
        print(f"Win type: {result['win_type']}")
        print(f"Pot: ${result['pot']:.0f}")
        if result['win_type'] == 'showdown':
            print(f"Winning hand: {result['winning_hand']}")
        break
```

## Features

### ✅ Complete Texas Hold'em Rules
- Blinds (small/big)
- Preflop, flop, turn, river betting rounds
- Showdown with hand evaluation
- Side pots (for all-ins)
- Proper position handling

### ✅ Action Validation
- Checks legal actions for each player
- Handles all-in situations
- Enforces minimum raise sizes
- Prevents invalid actions

### ✅ State Management
- Full game state tracking
- Action history recording
- Player position management
- Pot calculations

### ✅ Reusable Design
- Works for PPO training (automated play)
- Works for demo interface (human interaction)
- Supports 2-10 players
- Configurable blinds and stack sizes

## Integration

### PPO Training
```python
# In ppo/train_ppo.py
from poker_game import PokerGame, Action

game = PokerGame(num_players=2)

for episode in range(num_episodes):
    state = game.reset()
    
    while True:
        # Get action from AI agent
        action, amount = agent.get_action(state)
        
        # Execute in environment
        new_state, done, result = game.step(action, amount)
        
        # Store experience for PPO
        experiences.append((state, action, reward, new_state, done))
        
        if done:
            break
```

### Demo Interface
```python
# In demo/app.py
from poker_game import PokerGame, Action
import gradio as gr

game = PokerGame(num_players=2)

def play_hand(user_action):
    state, done, result = game.step(user_action)
    
    if not done:
        # AI's turn
        ai_action = ai_agent.get_action(state)
        state, done, result = game.step(ai_action)
    
    return game.get_game_state_string(), done, result
```

## Testing

Run tests to verify game logic:

```bash
python -m pytest poker_game/tests/
```

## Future Enhancements

- [ ] Side pot handling for multiple all-ins
- [ ] Tournament mode (increasing blinds)
- [ ] Multi-way pot splitting
- [ ] Poker hand history export
- [ ] Monte Carlo hand strength estimation
- [ ] Equity calculations

## Dependencies

- Python 3.8+
- No external dependencies (uses only standard library)

## License

MIT License - Free to use for academic and commercial projects
