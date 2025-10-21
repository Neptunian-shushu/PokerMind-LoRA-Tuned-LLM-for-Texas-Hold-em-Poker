// Backend(ai service) API URL
const API_URL = 'http://localhost:3001';

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
The player positions involved in this game are: ${gameState.players.map(p => p.name).join(', ')}.
In this hand, your position is ${position}, and your holding is [${hand}].

Before the flop:
${actionHistory}

Now it is your turn to make a move.
To remind you, the current pot size is ${gameState.pot} chips, and your holding is [${hand}].
Your remaining chips: ${player.chips} chips.
${toCall > 0 ? `You need to call ${toCall} chips to stay in the hand.` : 'You can check or bet.'}

Available Actions:
${toCall === 0 ? '- CHECK (cost: 0 chips)' : '- CALL ' + toCall + ' chips'}
${toCall > 0 ? '- FOLD (forfeit the hand)' : ''}
- RAISE (minimum raise: ${gameState.bigBlind} chips on top of the call amount)
- ALL-IN (bet all ${player.chips} remaining chips)

Please respond with ONLY a JSON object in the following format:
{
  "action": "FOLD|CHECK|CALL|RAISE|ALL-IN",
  "raiseAmount": <number if action is RAISE, 0 otherwise>,
  "reasoning": "<brief explanation of your decision based on hand strength, position, pot odds, and opponent actions>",
  "confidence": <number 0-100>
}

Consider factors like:
- Hand strength in 6-handed play
- Your position relative to the button
- Pot odds and implied odds
- Stack-to-pot ratio
- Opponent betting patterns and tendencies
- Expected value of each action

Respond ONLY with the JSON object, no other text.`;

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
  gameState.players.forEach((p, idx) => {
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

const extractJSON = (content) => {
  const jsonMatch = content.match(/```(?:json)?\s*\n?([\s\S]*?)\n?```/);
  if (jsonMatch) {
    return jsonMatch[1].trim();
  }
  return content.trim();
};

// call backend API to get AI decision
export const getAIDecision = async (gameState, playerId) => {
  try {
    const prompt = buildPreflopStatePrompt(gameState, playerId);
    
    console.log('Calling backend API for AI decision...');
    
    const response = await fetch(`${API_URL}/api/ai-decision`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ prompt })
    });

    if (!response.ok) {
      throw new Error(`API request failed: ${response.status}`);
    }

    const data = await response.json();
    const content = data.content;
    
    const jsonContent = extractJSON(content);
    const decision = JSON.parse(jsonContent);
    
    console.log('AI Decision:', decision);
    
    return {
      action: decision.action.toLowerCase().replace('-', ''),
      raiseAmount: decision.raiseAmount || 0,
      reasoning: decision.reasoning,
      confidence: decision.confidence
    };
    
  } catch (error) {
    console.error('Error calling backend API:', error);
    console.log('Falling back to rule-based AI...');
  }
};

export default {
  getAIDecision,
};

