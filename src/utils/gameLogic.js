const SUITS = ['hearts', 'diamonds', 'clubs', 'spades'];
const RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'];

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

// initialize game state
export const initializeGame = (playerNames, startingChips, smallBlind, bigBlind) => {
  const deck = shuffleDeck(createDeck());
  const { playerHands, remainingDeck } = dealCards(deck, playerNames.length);

  // determine positions - randomly select dealer
  const dealerPosition = Math.floor(Math.random() * playerNames.length); // random dealer position
  const sbPosition = (dealerPosition + 1) % playerNames.length; // small blind position
  const bbPosition = (dealerPosition + 2) % playerNames.length; // big blind position
  const firstToAct = (bbPosition + 1) % playerNames.length; // first to act player
  
  const players = playerNames.map((name, index) => {
    let initialBet = 0;
    let initialChips = startingChips;
    
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
      cards: playerHands[index].map((card, cardIndex) => ({
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
  
  if (activePlayers.length === 1) {
    return true; // only one player left, round ends
  }
  
  // all active players have acted
  const allActed = activePlayers.every(p => p.action !== '' && p.action !== 'SB' && p.action !== 'BB');
  
  // all active players' bets are the same
  const allBetsEqual = activePlayers.every(p => p.totalBetThisRound === gameState.currentBet);
  
  return allActed && allBetsEqual;
};

// player action
export const playerAction = (gameState, action, raiseAmount = 0) => {
  const newState = { ...gameState };
  const player = { ...newState.players[newState.currentPlayerIndex] };
  
  switch (action) {
    case 'fold':
      player.isFolded = true;
      player.action = 'Fold';
      break;
      
    case 'call':
      const callAmount = newState.currentBet - player.totalBetThisRound;
      player.chips -= callAmount;
      player.bet = callAmount;
      player.totalBetThisRound += callAmount;
      newState.pot += callAmount;
      player.action = 'Call';
      break;
      
    case 'check':
      player.action = 'Check';
      break;
      
    case 'raise':
      const totalRaise = raiseAmount;
      const toCall = newState.currentBet - player.totalBetThisRound;
      const totalCost = toCall + totalRaise;
      
      player.chips -= totalCost;
      player.bet = totalCost;
      player.totalBetThisRound += totalCost;
      newState.pot += totalCost;
      newState.currentBet = player.totalBetThisRound;
      newState.lastRaiserIndex = newState.currentPlayerIndex;
      player.action = `Raise ${totalRaise}`;
      
      // reset other players' action status (except for folded players)
      newState.players.forEach((p, idx) => {
        if (idx !== newState.currentPlayerIndex && !p.isFolded) {
          newState.players[idx] = { ...p };
        }
      });
      break;
      
    case 'all-in':
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
      break;
  }
  
  newState.players[newState.currentPlayerIndex] = player;
  newState.playersActedCount++;
  
  return newState;
};

// move to next player
export const moveToNextPlayer = (gameState) => {
  const newState = { ...gameState };
  let nextIndex = (newState.currentPlayerIndex + 1) % newState.players.length;
  
  // skip folded players
  while (newState.players[nextIndex].isFolded) {
    nextIndex = (nextIndex + 1) % newState.players.length;
  }
  
  newState.currentPlayerIndex = nextIndex;
  
  // reset current player's bet display
  newState.players = newState.players.map(p => ({ ...p, bet: 0 }));
  
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
  } else if (player.chips >= toCall) {
    actions.push('call');
  }
  
  if (player.chips > toCall) {
    actions.push('raise');
  }
  
  if (player.chips > 0) {
    actions.push('all-in');
  }
  
  return {
    actions,
    toCall,
    minRaise: gameState.bigBlind,
    maxRaise: player.chips - toCall
  };
};

// 格式化手牌字符串
export const formatHand = (cards) => {
  return cards.map(card => `${card.rank}${getSuitSymbol(card.suit)}`).join(' ');
};

// 获取花色符号
const getSuitSymbol = (suit) => {
  const symbols = {
    hearts: '♥',
    diamonds: '♦',
    clubs: '♣',
    spades: '♠'
  };
  return symbols[suit] || '';
};

// 确定游戏结果
export const determineWinner = (gameState) => {
  const activePlayers = gameState.players.filter(p => !p.isFolded);
  
  // 情况1：只剩一个玩家（其他人都弃牌了），Preflop 结束
  if (activePlayers.length === 1) {
    return {
      isFinished: true,           
      winnerId: activePlayers[0].id,  
      reason: 'all_folded',      
      gameState: gameState       
    };
  }
  
  // 情况2：多个玩家还在游戏中，需要继续到 Postflop
  return {
    isFinished: false,           
    winnerId: null,               
    reason: 'continue_to_postflop', 
    gameState: gameState,         
    activePlayers: activePlayers  
  };
};

