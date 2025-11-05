const SUITS = ['hearts', 'diamonds', 'clubs', 'spades'];
const RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'];

export const BETTING_ROUNDS = {
  PREFLOP: 'preflop',
  FLOP: 'flop',
  TURN: 'turn',
  RIVER: 'river',
  SHOWDOWN: 'showdown'
};

// create a new deck
export const createDeck = () => {
  const deck = [];
  for (const suit of SUITS) {
    for (const rank of RANKS) {
      deck.push({ rank, suit, faceDown: false });
    }
  }
  return deck;
};

// shuffle the deck
export const shuffleDeck = (deck) => {
  const shuffled = [...deck];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
};

// deal cards
export const dealCards = (deck, numPlayers) => {
  const playerHands = Array(numPlayers).fill(null).map(() => []);
  let cardIndex = 0;
  
  // deal two cards to each player
  for (let round = 0; round < 2; round++) {
    for (let player = 0; player < numPlayers; player++) {
      playerHands[player].push(deck[cardIndex++]);
    }
  }
  
  return { 
    playerHands, 
    remainingDeck: deck.slice(cardIndex) 
  };
};

// deal community cards
export const dealCommunityCards = (deck, count) => {
  // Burn a card first (optional, but traditional)
  deck.shift();
  return deck.splice(0, count);
};

// initialize game state
export const initializeGame = (playerNames, startingChips, smallBlind, bigBlind, previousDealerPosition = null) => {
  const deck = shuffleDeck(createDeck());
  const { playerHands, remainingDeck } = dealCards(deck, playerNames.length);

  // determine positions - first game: random dealer, subsequent games: rotate clockwise
  let dealerPosition;
  if (previousDealerPosition === null) {
    // First game: random dealer position
    dealerPosition = Math.floor(Math.random() * playerNames.length);
  } else {
    // Subsequent games: rotate clockwise by one position
    dealerPosition = (previousDealerPosition + 1) % playerNames.length;
  }
  const sbPosition = (dealerPosition + 1) % playerNames.length; // small blind position
  const bbPosition = (dealerPosition + 2) % playerNames.length; // big blind position
  const firstToAct = (bbPosition + 1) % playerNames.length; // first to act player
  
  // Support both single value or array for starting chips
  const chipsArray = Array.isArray(startingChips) ? startingChips : Array(playerNames.length).fill(startingChips);
  
  const players = playerNames.map((name, index) => {
    let initialBet = 0;
    let initialChips = chipsArray[index] || chipsArray[0] || 100;
    
    if (index === sbPosition) {
      initialBet = smallBlind;
      initialChips -= smallBlind;
    } else if (index === bbPosition) {
      initialBet = bigBlind;
      initialChips -= bigBlind;
    }
    
    return {
      id: index,
      name,
      chips: initialChips,
      cards: playerHands[index].map((card) => ({
        ...card,
        // only the player himself (index 0) can see his cards
        faceDown: index !== 0
      })),
      isActive: true,
      isFolded: false,
      isHuman: index === 0,
      position: getPositionName(index, playerNames.length),
      bet: initialBet,
      action: initialBet > 0 ? (initialBet === smallBlind ? 'SB' : 'BB') : '',
      lastAction: '', // Store last round's action for display
      isDealer: index === dealerPosition,
      totalBetThisRound: initialBet
    };
  });
  
  return {
    players,
    deck: remainingDeck,
    pot: smallBlind + bigBlind,
    currentBet: bigBlind,
    currentPlayerIndex: firstToAct,
    dealerPosition,
    smallBlind,
    bigBlind,
    phase: 'preflop',
    communityCards: [],
    lastRaiserIndex: bbPosition, // big blind is the initial "raiser"
    playersActedCount: 0
  };
};

// get position name (for display)
const getPositionName = (index, totalPlayers) => {
  if (totalPlayers === 6) {
    const positions = ['bottom', 'left-bottom', 'left-top', 'top', 'right-top', 'right-bottom'];
    return positions[index] || 'bottom';
  }
  return 'bottom';
};

// check if all players have acted and bet the same amount
export const isRoundComplete = (gameState) => {
  const activePlayers = gameState.players.filter(p => !p.isFolded);
  
  // If only one player remains, round is complete
  if (activePlayers.length === 1) {
    return true;
  }
  
  // Players who can still act (have chips to bet)
  const playersWithChips = activePlayers.filter(p => p.chips > 0);
  
  // If no players have chips left (all all-in), round is complete
  if (playersWithChips.length === 0) {
    return true;
  }
  
  // If only one player has chips and others are all-in, check if they acted
  if (playersWithChips.length === 1) {
    return playersWithChips[0].action !== '' && playersWithChips[0].action !== 'SB' && playersWithChips[0].action !== 'BB';
  }
  
  // Check if all players with chips have acted
  const allActed = playersWithChips.every(p => p.action !== '' && p.action !== 'SB' && p.action !== 'BB');
  
  // Check if all bets are settled among players with chips
  const allBetsSettled = playersWithChips.every(p => p.totalBetThisRound === gameState.currentBet);
  
  return allActed && allBetsSettled;
};

// player action
export const playerAction = (gameState, action, raiseAmount = 0) => {
  const newState = { 
    ...gameState,
    players: [...gameState.players]  // create a new array to ensure React detects changes
  };
  const player = { ...newState.players[newState.currentPlayerIndex] };
  
  switch (action) {
    case 'fold':
      player.isFolded = true;
      player.action = 'Fold';
      break;
      
    case 'call':
      {
        const callAmount = newState.currentBet - player.totalBetThisRound;
        player.chips -= callAmount;
        player.bet = callAmount;
        player.totalBetThisRound += callAmount;
        newState.pot += callAmount;
        player.action = 'Call';
        
        // In preflop, preserve SB/BB actions for other players who haven't acted
        if (newState.phase === 'preflop') {
          newState.players.forEach((p, idx) => {
            if (idx !== newState.currentPlayerIndex && !p.isFolded) {
              // If action is already SB or BB, keep it
              if (p.action === 'SB' || p.action === 'BB') {
                // Keep the existing blind action
                newState.players[idx] = { ...p };
              }
              // If action is empty/undefined but player has blind bet, restore it
              else if (p.action === '' || p.action === undefined) {
                if (p.totalBetThisRound === newState.smallBlind) {
                  newState.players[idx] = { ...p, action: 'SB' };
                } else if (p.totalBetThisRound === newState.bigBlind) {
                  newState.players[idx] = { ...p, action: 'BB' };
                }
              }
            }
          });
        }
      }
      break;
      
    case 'check':
      player.action = 'Check';
      
      // In preflop, preserve SB/BB actions for other players who haven't acted
      if (newState.phase === 'preflop') {
        newState.players.forEach((p, idx) => {
          if (idx !== newState.currentPlayerIndex && !p.isFolded) {
            // If action is already SB or BB, keep it
            if (p.action === 'SB' || p.action === 'BB') {
              // Keep the existing blind action
              newState.players[idx] = { ...p };
            }
            // If action is empty/undefined but player has blind bet, restore it
            else if (p.action === '' || p.action === undefined) {
              if (p.totalBetThisRound === newState.smallBlind) {
                newState.players[idx] = { ...p, action: 'SB' };
              } else if (p.totalBetThisRound === newState.bigBlind) {
                newState.players[idx] = { ...p, action: 'BB' };
              }
            }
          }
        });
      }
      break;
      
    case 'bet':
    case 'raise':
      {
        // raiseAmount/betAmount is the target total bet amount (bet/raise to)
        // e.g., bet 3 or raise 2.5 means "bet/raise to 3/2.5 total"
        const newTotalBet = raiseAmount;
        const isBet = action === 'bet';
        
        // Validate: bet/raise must be greater than current bet
        if (newTotalBet <= newState.currentBet) {
          throw new Error(`${isBet ? 'Bet' : 'Raise'} amount must be greater than current bet of ${newState.currentBet}`);
        }
        
        // Validate: player must have enough chips
        const totalCost = newTotalBet - player.totalBetThisRound;
        if (totalCost > player.chips) {
          throw new Error(`Insufficient chips. Need ${totalCost}, have ${player.chips}`);
        }
        
        player.chips -= totalCost;
        player.bet = totalCost;
        player.totalBetThisRound = newTotalBet;
        newState.pot += totalCost;
        newState.currentBet = newTotalBet;
        newState.lastRaiserIndex = newState.currentPlayerIndex;
        player.action = isBet ? `Bet ${newTotalBet}` : `Raise ${newTotalBet}`;
        
        // reset other players' action status (except for folded players)
        // In preflop, preserve SB/BB actions for players who haven't acted yet
        newState.players.forEach((p, idx) => {
          if (idx !== newState.currentPlayerIndex && !p.isFolded) {
            const updatedPlayer = { ...p };
            // In preflop, preserve or restore blind actions
            if (newState.phase === 'preflop') {
              // If action is already SB or BB, keep it
              if (p.action === 'SB' || p.action === 'BB') {
                // Keep the existing blind action
                updatedPlayer.action = p.action;
              } 
              // If action is empty/undefined but player has blind bet, restore it
              else if (p.action === '' || p.action === undefined) {
                if (p.totalBetThisRound === newState.smallBlind) {
                  updatedPlayer.action = 'SB';
                } else if (p.totalBetThisRound === newState.bigBlind) {
                  updatedPlayer.action = 'BB';
                }
              }
            }
            newState.players[idx] = updatedPlayer;
          }
        });
      }
      break;
      
    case 'all-in':
      {
        const allInAmount = player.chips;
        player.totalBetThisRound += allInAmount;
        newState.pot += allInAmount;
        player.bet = allInAmount;
        player.chips = 0;
        
        if (player.totalBetThisRound > newState.currentBet) {
          newState.currentBet = player.totalBetThisRound;
          newState.lastRaiserIndex = newState.currentPlayerIndex;
        }
        
        player.action = 'All-In';
      }
      break;
  }
  
  newState.players[newState.currentPlayerIndex] = player;
  newState.playersActedCount++;
  
  return newState;
};

// move to next player
export const moveToNextPlayer = (gameState) => {
  let nextIndex = (gameState.currentPlayerIndex + 1) % gameState.players.length;
  let attempts = 0;
  const maxAttempts = gameState.players.length;
  
  // skip folded players and all-in players (who have no chips left)
  while ((gameState.players[nextIndex].isFolded || gameState.players[nextIndex].chips === 0) && attempts < maxAttempts) {
    nextIndex = (nextIndex + 1) % gameState.players.length;
    attempts++;
  }
  
  // create new state with new players array and reset bet display
  const newState = {
    ...gameState,
    currentPlayerIndex: nextIndex,
    players: gameState.players.map((p, idx) => {
      const updatedPlayer = { ...p, bet: 0 };
      
      // Save action as lastAction for the player who just acted (previous current player)
      // Only if they're not the next current player and have a non-blind action
      if (idx === gameState.currentPlayerIndex && p.action && p.action !== 'SB' && p.action !== 'BB') {
        updatedPlayer.lastAction = p.action;
      }
      
      // In preflop, preserve SB/BB actions for players who haven't acted yet
      if (gameState.phase === 'preflop' && (p.action === 'SB' || p.action === 'BB')) {
        // Keep the blind action
        updatedPlayer.action = p.action;
      } else if (gameState.phase === 'preflop' && (p.action === '' || p.action === undefined) && !p.isFolded) {
        // Restore blind action if player hasn't acted
        if (p.totalBetThisRound === gameState.smallBlind) {
          updatedPlayer.action = 'SB';
        } else if (p.totalBetThisRound === gameState.bigBlind) {
          updatedPlayer.action = 'BB';
        }
      }
      return updatedPlayer;
    })
  };
  
  return newState;
};

// get available actions for current player
export const getAvailableActions = (gameState) => {
  const player = gameState.players[gameState.currentPlayerIndex];
  const toCall = gameState.currentBet - player.totalBetThisRound;
  
  const actions = [];
  
  if (toCall > 0) {
    actions.push('fold');
  }

  if (toCall === 0) {
    actions.push('check');
    // When no one has bet, player can bet
    if (player.chips > 0) {
      actions.push('bet');
    }
  } else if (player.chips >= toCall) {
    actions.push('call');
  }
  
  if (player.chips > toCall) {
    actions.push('raise');
  }
  
  if (player.chips > 0) {
    actions.push('all-in');
  }
  
  // minRaise: minimum total bet amount (must be at least currentBet + 1)
  // maxRaise: maximum total bet amount (currentBet + remaining chips)
  const minRaise = gameState.currentBet + 1;
  const maxRaise = gameState.currentBet + player.chips;
  
  return {
    actions,
    toCall,
    minRaise,
    maxRaise
  };
};

// Format hand string
export const formatHand = (cards) => {
  return cards.map(card => `${card.rank}${getSuitSymbol(card.suit)}`).join(' ');
};

// Get suit symbol
const getSuitSymbol = (suit) => {
  const symbols = {
    hearts: '♥',
    diamonds: '♦',
    clubs: '♣',
    spades: '♠'
  };
  return symbols[suit] || '';
};

// Advance to next betting round
export const advanceToNextRound = (gameState) => {
  // Reset player states, but preserve last action for display (only for active players)
  const resetPlayers = gameState.players.map(p => {
    // Save current action as lastAction if player is still active (not folded) and has an action
    const newLastAction = !p.isFolded && p.action && p.action !== 'SB' && p.action !== 'BB' 
      ? p.action 
      : p.lastAction;
    
    return {
      ...p,
      bet: 0,
      totalBetThisRound: 0,
      lastAction: newLastAction,
      action: '' // Clear current action for new round
    };
  });
  
  const newState = { 
    ...gameState,
    players: resetPlayers
  };
  
  // Reset round state
  newState.currentBet = 0;
  newState.playersActedCount = 0;
  newState.lastRaiserIndex = -1;
  
  // Handle transitions between different betting rounds
  switch(newState.phase) {
    case BETTING_ROUNDS.PREFLOP:
      // Deal the flop
      newState.communityCards = dealCommunityCards(newState.deck, 3);
      newState.phase = BETTING_ROUNDS.FLOP;
      break;
    
    case BETTING_ROUNDS.FLOP:
      // Deal the turn
      newState.communityCards = [...newState.communityCards, ...dealCommunityCards(newState.deck, 1)];
      newState.phase = BETTING_ROUNDS.TURN;
      break;
    
    case BETTING_ROUNDS.TURN:
      // Deal the river
      newState.communityCards = [...newState.communityCards, ...dealCommunityCards(newState.deck, 1)];
      newState.phase = BETTING_ROUNDS.RIVER;
      break;
    
    case BETTING_ROUNDS.RIVER:
      // After river round, proceed to showdown and reveal all active players' cards
      newState.phase = BETTING_ROUNDS.SHOWDOWN;
      newState.players = newState.players.map(player => {
        if (!player.isFolded) {
          return {
            ...player,
            cards: player.cards.map(card => ({ ...card, faceDown: false }))
          };
        }
        return player;
      });
      break;
  }
  
  // Set first player to act (starting from position after dealer)
  let nextToAct = (newState.dealerPosition + 1) % newState.players.length;
  let attempts = 0;
  const maxAttempts = newState.players.length;
  
  // Skip folded players and all-in players (who have no chips left to act)
  while ((newState.players[nextToAct].isFolded || newState.players[nextToAct].chips === 0) && attempts < maxAttempts) {
    nextToAct = (nextToAct + 1) % newState.players.length;
    attempts++;
  }
  newState.currentPlayerIndex = nextToAct;
  
  return newState;
};

// Get hand description
const getHandDescription = (cards) => {
  return cards.map(card => `${card.rank}${getSuitSymbol(card.suit)}`).join(' ');
};

// Determine game result
export const determineWinner = (gameState) => {
  const activePlayers = gameState.players.filter(p => !p.isFolded);
  
  // Create player result statistics
  const playerStats = activePlayers.map(player => {
    const netGain = -player.totalBetThisRound; // Initially negative (amount invested)
    return {
      ...player,
      netGain,
      handDescription: getHandDescription(player.cards)
    };
  });
  
  // Case 1: Only one player left (all others folded)
  if (activePlayers.length === 1) {
    const winnerId = activePlayers[0].id;
    // Update all players' cards to visible and update winner's chips
    const updatedPlayers = gameState.players.map(player => {
      const playerCopy = {
        ...player,
        cards: player.cards.map(card => ({ ...card, faceDown: false }))
      };
      // Winner gets the pot
      if (player.id === winnerId) {
        playerCopy.chips += gameState.pot;
      }
      return playerCopy;
    });
    
    // Update winner's net gain
    playerStats[0].netGain += gameState.pot;

    return {
      isFinished: true,
      winnerId: winnerId,
      reason: 'all_folded',
      gameState: { ...gameState, players: updatedPlayers },
      playerStats,
      summary: {
        winType: 'fold',
        potSize: gameState.pot,
        handDescriptions: playerStats.map(p => ({
          name: p.name,
          cards: p.handDescription,
          netGain: p.netGain
        }))
      }
    };
  }
  
  // Case 2: If showdown, need to compare best 5 cards from 7 cards
  if (gameState.phase === BETTING_ROUNDS.SHOWDOWN) {
    // All players reveal cards
    const updatedPlayers = gameState.players.map(player => ({
      ...player,
      cards: player.cards.map(card => ({ ...card, faceDown: false }))
    }));

    // Evaluate each player's best five cards
    const handValues = {}; // id -> {score, name}
    activePlayers.forEach(player => {
      const sevenCards = [...player.cards, ...gameState.communityCards];
      const best = evaluateBestHandFromSeven(sevenCards);
      handValues[player.id] = best; // { score: number[], name: string }
    });

    // Find all winners (handle ties - multiple players with same best hand)
    const playerIds = Object.keys(handValues).map(id => parseInt(id));
    
    // Find the best score (highest hand value)
    const bestPlayerId = playerIds.reduce((best, id) => {
      const score = handValues[id].score;
      const currentBest = handValues[best].score;
      return compareScores(score, currentBest) > 0 ? id : best;
    }, playerIds[0]);
    
    const bestScore = handValues[bestPlayerId].score;
    
    // Find all players with the best score (ties)
    const winnerIds = playerIds.filter(id => {
      return compareScores(handValues[id].score, bestScore) === 0;
    });
    
    // Split pot equally among winners
    const potPerWinner = Math.floor(gameState.pot / winnerIds.length);
    const remainder = gameState.pot % winnerIds.length; // Handle remainder chips

    // Update winners' net gain
    winnerIds.forEach(winningId => {
      const winnerIndex = playerStats.findIndex(p => p.id === winningId);
      if (winnerIndex !== -1) {
        // Each winner gets their share of the pot
        const winnerShare = potPerWinner + (winnerIds.indexOf(winningId) < remainder ? 1 : 0);
        playerStats[winnerIndex].netGain += winnerShare;
      }
    });

    // Update winners' chips (split pot)
    const finalPlayers = updatedPlayers.map(player => {
      if (winnerIds.includes(player.id)) {
        const winnerShare = potPerWinner + (winnerIds.indexOf(player.id) < remainder ? 1 : 0);
        return {
          ...player,
          chips: player.chips + winnerShare
        };
      }
      return player;
    });
    
    // Primary winner ID (for compatibility with existing code)
    const primaryWinnerId = winnerIds[0];

    return {
      isFinished: true,
      winnerId: primaryWinnerId, // Primary winner for backward compatibility
      winnerIds: winnerIds, // All winners (for ties)
      isTie: winnerIds.length > 1,
      reason: winnerIds.length > 1 ? 'showdown_tie' : 'showdown',
      gameState: { ...gameState, players: finalPlayers },
      showdown: true,
      activePlayers: activePlayers,
      handValues,
      playerStats,
      summary: {
        winType: winnerIds.length > 1 ? 'showdown_tie' : 'showdown',
        potSize: gameState.pot,
        potPerWinner: potPerWinner,
        winnerCount: winnerIds.length,
        handDescriptions: playerStats.map(p => ({
          name: p.name,
          cards: p.handDescription,
          handValue: handValues[p.id] ? handValues[p.id].name : 'N/A',
          netGain: p.netGain,
          isWinner: winnerIds.includes(p.id)
        }))
      }
    };
  }

  // Case 3: Need to continue to next round
  return {
    isFinished: false,
    winnerId: null,
    reason: 'continue_to_next_round',
    gameState: gameState,
    activePlayers: activePlayers
  };
};

// ===== Texas Hold'em Hand Evaluation Helper Functions =====

// Convert rank character to numeric value
const rankToValue = (rank) => {
  if (rank === 'A') return 14;
  if (rank === 'K') return 13;
  if (rank === 'Q') return 12;
  if (rank === 'J') return 11;
  return parseInt(rank, 10);
};

// Combination utility: get all combinations of k elements from array
const combinations = (arr, k) => {
  const result = [];
  const helper = (start, combo) => {
    if (combo.length === k) {
      result.push([...combo]);
      return;
    }
    for (let i = start; i < arr.length; i++) {
      combo.push(arr[i]);
      helper(i + 1, combo);
      combo.pop();
    }
  };
  helper(0, []);
  return result;
};

// Compare scores: returns positive if a>b, negative if a<b, 0 if equal
const compareScores = (a, b) => {
  const len = Math.max(a.length, b.length);
  for (let i = 0; i < len; i++) {
    const av = a[i] ?? 0;
    const bv = b[i] ?? 0;
    if (av !== bv) return av - bv;
  }
  return 0;
};

// Evaluate 5 cards, returns { score: [...], name: string }
// Score structure:
// [category, main_value1, main_value2, kicker1, kicker2, kicker3]
// Category: 8=Straight Flush, 7=Four of a Kind, 6=Full House, 5=Flush, 4=Straight, 3=Three of a Kind, 2=Two Pair, 1=One Pair, 0=High Card
const evaluateFive = (cards5) => {
  const values = cards5.map(c => rankToValue(c.rank)).sort((a,b) => b - a);
  const suits = cards5.map(c => c.suit);

  // Handle A as 1 for straight detection
  const uniqueDesc = [...new Set(values)].sort((a,b) => b - a);
  const isWheel = uniqueDesc.includes(14) && uniqueDesc.includes(5) && uniqueDesc.includes(4) && uniqueDesc.includes(3) && uniqueDesc.includes(2);
  const straightHigh = (() => {
    // Regular straight
    if (uniqueDesc.length >= 5) {
      // With only 5 cards, uniqueDesc is either 5 or less (with duplicates)
      const arr = uniqueDesc;
      // Check if five consecutive from max to min
      if (arr.length === 5 && arr[0] - arr[4] === 4) return arr[0];
    }
    // A2345
    if (isWheel) return 5;
    return null;
  })();

  const suitCounts = suits.reduce((m, s) => (m[s] = (m[s]||0)+1, m), {});
  const isFlush = Object.values(suitCounts).some(c => c === 5);

  const countByValue = values.reduce((m, v) => (m[v] = (m[v]||0)+1, m), {});
  const groups = Object.entries(countByValue).map(([v,c]) => ({ v: parseInt(v,10), c }));
  // Sort by count descending, then by value descending
  groups.sort((a,b) => b.c - a.c || b.v - a.v);

  // Straight Flush
  if (isFlush && straightHigh) {
    return { score: [8, straightHigh], name: 'Straight Flush' };
  }

  // Four of a Kind
  if (groups[0]?.c === 4) {
    const four = groups[0].v;
    const kicker = groups.find(g => g.v !== four)?.v || 0;
    return { score: [7, four, kicker], name: 'Four of a Kind' };
  }

  // Full House
  if (groups[0]?.c === 3 && groups[1]?.c === 2) {
    return { score: [6, groups[0].v, groups[1].v], name: 'Full House' };
  }

  // Flush
  if (isFlush) {
    return { score: [5, ...values], name: 'Flush' };
  }

  // Straight
  if (straightHigh) {
    return { score: [4, straightHigh], name: 'Straight' };
  }

  // Three of a Kind
  if (groups[0]?.c === 3) {
    const trips = groups[0].v;
    const kickers = groups.filter(g => g.v !== trips).map(g => g.v).sort((a,b) => b - a);
    return { score: [3, trips, ...kickers.slice(0,2)], name: 'Three of a Kind' };
  }

  // Two Pair
  if (groups[0]?.c === 2 && groups[1]?.c === 2) {
    const highPair = Math.max(groups[0].v, groups[1].v);
    const lowPair = Math.min(groups[0].v, groups[1].v);
    const kicker = groups.find(g => g.c === 1 && g.v !== highPair && g.v !== lowPair)?.v || 0;
    return { score: [2, highPair, lowPair, kicker], name: 'Two Pair' };
    }

  // One Pair
  if (groups[0]?.c === 2) {
    const pair = groups[0].v;
    const kickers = groups.filter(g => g.v !== pair).map(g => g.v).sort((a,b) => b - a);
    return { score: [1, pair, ...kickers.slice(0,3)], name: 'One Pair' };
  }

  // High Card
  return { score: [0, ...values], name: 'High Card' };
};

// Select best 5 cards from 7 cards
const evaluateBestHandFromSeven = (cards7) => {
  // Get all 5-card combinations
  const all5 = combinations(cards7, 5);
  let best = null;
  for (const hand of all5) {
    const eval5 = evaluateFive(hand);
    if (!best || compareScores(eval5.score, best.score) > 0) {
      best = eval5;
    }
  }
  return best;
};

