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
  
  // 如果只剩一个玩家，回合立即结束
  if (activePlayers.length === 1) {
    return true;
  }
  
  // 检查所有活跃玩家是否都已行动（除了小盲和大盲的初始状态）
  const allActed = activePlayers.every(p => p.action !== '' && p.action !== 'SB' && p.action !== 'BB');
  
  // 检查所有活跃玩家的下注是否相同
  const allBetsEqual = activePlayers.every(p => p.totalBetThisRound === gameState.currentBet);
  // 当存在全下（筹码为0）的玩家时，允许该玩家不与当前最大下注量相等，也视作筹码结清
  const allBetsSettled = activePlayers.every(p => p.totalBetThisRound === gameState.currentBet || p.chips === 0);

  // 如果在河牌轮且所有玩家都check了（或者全都采取了行动且下注相同），强制进入摊牌
  const allChecked = activePlayers.every(p => p.action === 'Check');
  const isRiverRound = gameState.phase === BETTING_ROUNDS.RIVER;
  
  // 回合结束条件：
  // 1. 所有玩家都行动过且下注相同
  // 2. 在河牌轮且所有人都check
  return (allActed && (allBetsEqual || allBetsSettled)) || (isRiverRound && allChecked);
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
      {
        const callAmount = newState.currentBet - player.totalBetThisRound;
        player.chips -= callAmount;
        player.bet = callAmount;
        player.totalBetThisRound += callAmount;
        newState.pot += callAmount;
        player.action = 'Call';
      }
      break;
      
    case 'check':
      player.action = 'Check';
      break;
      
    case 'raise':
      {
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

// 推进到下一个回合
export const advanceToNextRound = (gameState) => {
  const newState = { ...gameState };
  
  // 重置玩家状态
  newState.players = newState.players.map(p => ({
    ...p,
    bet: 0,
    totalBetThisRound: 0,
    action: ''
  }));
  
  // 重置回合状态
  newState.currentBet = 0;
  newState.playersActedCount = 0;
  newState.lastRaiserIndex = -1;
  
  // 处理不同回合的转换
  switch(newState.phase) {
    case BETTING_ROUNDS.PREFLOP:
      // 发放翻牌
      newState.communityCards = dealCommunityCards(newState.deck, 3);
      newState.phase = BETTING_ROUNDS.FLOP;
      break;
    
    case BETTING_ROUNDS.FLOP:
      // 发放转牌
      newState.communityCards = [...newState.communityCards, ...dealCommunityCards(newState.deck, 1)];
      newState.phase = BETTING_ROUNDS.TURN;
      break;
    
    case BETTING_ROUNDS.TURN:
      // 发放河牌
      newState.communityCards = [...newState.communityCards, ...dealCommunityCards(newState.deck, 1)];
      newState.phase = BETTING_ROUNDS.RIVER;
      break;
    
    case BETTING_ROUNDS.RIVER:
      // 河牌轮结束后直接进入摊牌，并确保所有未弃牌的玩家亮出手牌
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
  
  // 设置第一个行动玩家（从庄家下一位开始）
  let nextToAct = (newState.dealerPosition + 1) % newState.players.length;
  while (newState.players[nextToAct].isFolded) {
    nextToAct = (nextToAct + 1) % newState.players.length;
  }
  newState.currentPlayerIndex = nextToAct;
  
  return newState;
};

// 获取玩家手牌的描述
const getHandDescription = (cards) => {
  return cards.map(card => `${card.rank}${getSuitSymbol(card.suit)}`).join(' ');
};

// 获取手牌的大小描述
const getHandValueDescription = (value) => {
  if (value >= 25) return '高牌对';
  if (value >= 20) return '中等对子';
  if (value >= 15) return '小对子';
  return '高牌';
};

// 确定游戏结果
export const determineWinner = (gameState) => {
  const activePlayers = gameState.players.filter(p => !p.isFolded);
  
  // 创建玩家结果统计
  const playerStats = activePlayers.map(player => {
    const netGain = -player.totalBetThisRound; // 初始为负投入金额
    return {
      ...player,
      netGain,
      handDescription: getHandDescription(player.cards)
    };
  });
  
  // 情况1：只剩一个玩家（其他人都弃牌了）
  if (activePlayers.length === 1) {
    const winnerId = activePlayers[0].id;
    // 更新所有玩家的手牌为可见
    const updatedPlayers = gameState.players.map(player => ({
      ...player,
      cards: player.cards.map(card => ({ ...card, faceDown: false }))
    }));
    
    // 更新获胜者的净收益
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
  
  // 情况2：如果是 showdown，需要比较7张牌中的最佳5张
  if (gameState.phase === BETTING_ROUNDS.SHOWDOWN) {
    // 所有玩家亮牌
    const updatedPlayers = gameState.players.map(player => ({
      ...player,
      cards: player.cards.map(card => ({ ...card, faceDown: false }))
    }));

    // 评估每位玩家的最佳五张牌
    const handValues = {}; // id -> {score, name}
    activePlayers.forEach(player => {
      const sevenCards = [...player.cards, ...gameState.communityCards];
      const best = evaluateBestHandFromSeven(sevenCards);
      handValues[player.id] = best; // { score: number[], name: string }
    });

    // 找到获胜者（按评分逐项比较）
    const winnerIdStr = Object.keys(handValues).reduce((bestId, currId) => {
      const a = handValues[bestId].score;
      const b = handValues[currId].score;
      return compareScores(b, a) > 0 ? currId : bestId; // 返回评分更高者
    }, Object.keys(handValues)[0]);
    const winningId = parseInt(winnerIdStr);

    // 更新获胜者的净收益
    const winnerIndex = playerStats.findIndex(p => p.id === winningId);
    if (winnerIndex !== -1) {
      playerStats[winnerIndex].netGain += gameState.pot;
    }

    return {
      isFinished: true,
      winnerId: winningId,
      reason: 'showdown',
      gameState: { ...gameState, players: updatedPlayers },
      showdown: true,
      activePlayers: activePlayers,
      handValues,
      playerStats,
      summary: {
        winType: 'showdown',
        potSize: gameState.pot,
        handDescriptions: playerStats.map(p => ({
          name: p.name,
          cards: p.handDescription,
          handValue: handValues[p.id] ? handValues[p.id].name : 'N/A',
          netGain: p.netGain
        }))
      }
    };
  }

  // 情况3：需要继续到下一轮
  return {
    isFinished: false,
    winnerId: null,
    reason: 'continue_to_next_round',
    gameState: gameState,
    activePlayers: activePlayers
  };
};

// ===== 德州手牌评估辅助函数 =====

// 将牌点字符转成数值
const rankToValue = (rank) => {
  if (rank === 'A') return 14;
  if (rank === 'K') return 13;
  if (rank === 'Q') return 12;
  if (rank === 'J') return 11;
  return parseInt(rank, 10);
};

// 组合工具：从数组中选取k个元素的所有组合
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

// 比较评分：返回正数表示a>b，负数表示a<b，0表示相等
const compareScores = (a, b) => {
  const len = Math.max(a.length, b.length);
  for (let i = 0; i < len; i++) {
    const av = a[i] ?? 0;
    const bv = b[i] ?? 0;
    if (av !== bv) return av - bv;
  }
  return 0;
};

// 评估5张牌，返回 { score: [...], name: string }
// 评分结构：
// [类别, 主值1, 主值2, kicker1, kicker2, kicker3]
// 类别：8=Straight Flush, 7=Four of a Kind, 6=Full House, 5=Flush, 4=Straight, 3=Three of a Kind, 2=Two Pair, 1=One Pair, 0=High Card
const evaluateFive = (cards5) => {
  const values = cards5.map(c => rankToValue(c.rank)).sort((a,b) => b - a);
  const suits = cards5.map(c => c.suit);

  // 处理A当作1用于顺子判断
  const uniqueDesc = [...new Set(values)].sort((a,b) => b - a);
  const isWheel = uniqueDesc.includes(14) && uniqueDesc.includes(5) && uniqueDesc.includes(4) && uniqueDesc.includes(3) && uniqueDesc.includes(2);
  const straightHigh = (() => {
    // 常规顺子
    if (uniqueDesc.length >= 5) {
      // 因为只有5张，uniqueDesc要么5要么更少（有重复）
      const arr = uniqueDesc;
      // 检查从最大到最小是否连五个
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
  // 先按数量降序，再按点数降序
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

// 从7张牌中选最佳5张
const evaluateBestHandFromSeven = (cards7) => {
  // 选取所有5张组合
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

