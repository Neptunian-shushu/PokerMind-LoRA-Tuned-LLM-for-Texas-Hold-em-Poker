// Backend(ai service) API URL
const API_BASE = import.meta.env.VITE_API_URL;

// construct prompt
const buildPreflopStatePrompt = (gameState, playerId) => {
  const player = gameState.players[playerId];
  
  // get hand information (formatted more clearly)
  const hand = player.cards.map(card => `${card.rank} of ${formatSuit(card.suit)}`).join(' and ');
  
  const position = getPositionFromIndex(playerId, gameState.dealerPosition, gameState.players.length);
  
  // build action history
  const actionHistory = buildActionHistory(gameState, playerId);
  
  const toCall = gameState.currentBet - player.totalBetThisRound;
  const startingChips = player.chips + player.totalBetThisRound;
  
  const prompt = `You are a specialist in playing 6-handed No Limit Texas Hold'em. The following will be a game scenario and you need to make the optimal decision.

Here is a game summary:

The small blind is ${gameState.smallBlind} chips and the big blind is ${gameState.bigBlind} chips. Everyone started with ${startingChips} chips.
In this hand, your position is ${position}, and your holding is [${hand}].

Before the flop:
${actionHistory}

Now it is your turn to make a move.
To remind you, the current pot size is ${gameState.pot} chips, and your holding is [${hand}].
Your remaining chips: ${player.chips} chips.
${toCall > 0 ? `You need to call ${toCall} chips to stay in the hand.` : 'You can check or bet.'}

Decide on an action based on the strength of your hand on this board, your position, and actions before you. Do not explain your answer.
Your optimal action is:
`;

  return prompt;
};

// get position name (6-handed standard positions)
const getPositionFromIndex = (playerIndex, dealerIndex, totalPlayers) => {
  const relativePos = (playerIndex - dealerIndex + totalPlayers) % totalPlayers;
  
  if (totalPlayers === 6) {
    // 6-handed standard positions
    const positions = ['BTN', 'SB', 'BB', 'UTG', 'HJ', 'CO'];
    return positions[relativePos] || `Position ${relativePos}`;
  } else {
    // other number of players' table
    const positions = ['Button', 'Small Blind', 'Big Blind', 'UTG'];
    return positions[relativePos] || `Position ${relativePos}`;
  }
};

// format suit name
const formatSuit = (suit) => {
  const suitNames = {
    hearts: 'Heart',
    diamonds: 'Diamond',
    clubs: 'Club',
    spades: 'Spade'
  };
  return suitNames[suit] || suit;
};

// build action history
const buildActionHistory = (gameState, playerId) => {
  const actions = [];
  
  // record blind bets
  gameState.players.forEach((p) => {
    if (p.action === 'SB') {
      actions.push(`${p.name} posted small blind (${gameState.smallBlind} chips)`);
    } else if (p.action === 'BB') {
      actions.push(`${p.name} posted big blind (${gameState.bigBlind} chips)`);
    }
  });
  
  // record other players' actions (in order)
  gameState.players.forEach((p, idx) => {
    if (idx !== playerId && p.action && p.action !== 'SB' && p.action !== 'BB') {
      if (p.isFolded) {
        actions.push(`${p.name} fold`);
      } else {
        actions.push(`${p.name} ${p.action.toLowerCase()}`);
      }
    }
  });
  
  return actions.length > 0 ? actions.join(', then ') + '.' : 'No actions yet.';
};

// call backend API to get AI decision
export const getAIDecision = async (gameState, playerId) => {
  try {
    const prompt = buildPreflopStatePrompt(gameState, playerId);
    //

    const response = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ instruction: prompt })
    });

    if (!response.ok) {
      throw new Error(`API request failed: ${response.status}`);
    }

    const data = await response.json();

    const action = data.action;
    const raiseAmount = Number(data.raiseAmount) || 0;

    console.log('AI Decision:', { action, raiseAmount });

    return {
      action,
      raiseAmount
    };
  } catch (error) {
    console.error('Error calling backend API:', error);
    alert('Backend server is not connected, please retry.');
    throw new Error('Backend connection failed');
  }
};

export default {
  getAIDecision,
};

