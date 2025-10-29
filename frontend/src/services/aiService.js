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

Please respond with ONLY a JSON object in the following format:
{
  "action": "FOLD|CHECK|CALL|RAISE|ALL-IN",
  "raiseAmount": <number if action is RAISE, 0 otherwise>
}
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

const extractJSON = (content) => {
  const jsonMatch = content.match(/```(?:json)?\s*\n?([\s\S]*?)\n?```/);
  if (jsonMatch) {
    return jsonMatch[1].trim();
  }
  return content.trim();
};

// call backend API to get AI decision
export const getAIDecision = async (gameState, playerId) => {
  // Helper: normalize action strings from the model to our internal action keys
  const normalizeAction = (raw) => {
    const a = String(raw || '').trim().toLowerCase();
    if (a === 'fold') return 'fold';
    if (a === 'check') return 'check';
    if (a === 'call') return 'call';
    if (a === 'raise') return 'raise';
    if (a === 'all-in' || a === 'all in' || a === 'allin') return 'all-in';
    // Default safe choice when unknown
    return 'check';
  };

  // Simple rule-based fallback when API is unavailable or returns invalid data
  const fallbackDecision = () => {
    const player = gameState.players[playerId];
    const toCall = Math.max(0, gameState.currentBet - player.totalBetThisRound);
    // If cannot cover the call, go all-in (short call)
    if (toCall > 0 && player.chips <= toCall) {
      return { action: 'all-in', raiseAmount: 0, reasoning: 'Short stack all-in to call', confidence: 60 };
    }
    // If nothing to call, check most of the time
    if (toCall === 0) {
      return { action: 'check', raiseAmount: 0, reasoning: 'Free check', confidence: 70 };
    }
    // Cheap call threshold
    const cheap = player.chips * 0.05;
    if (toCall <= cheap) {
      return { action: 'call', raiseAmount: 0, reasoning: 'Price is cheap to continue', confidence: 55 };
    }
    // Otherwise fold conservatively
    return { action: 'fold', raiseAmount: 0, reasoning: 'Conservative fold vs large bet', confidence: 55 };
  };

  try {
    const prompt = buildPreflopStatePrompt(gameState, playerId);
    console.log('Prompt:', prompt);

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
    console.log('AI Response Data:', data);
    const content = data.predicted_action || '';

    console.log('AI Response:', content);

    const jsonContent = extractJSON(content);
    const parsed = JSON.parse(jsonContent);

    const action = normalizeAction(parsed.action);
    const raiseAmount = Number(parsed.raiseAmount) || 0;

    console.log('AI Decision:', { action: parsed.action, normalized: action, raiseAmount });

    return {
      action,
      raiseAmount,
      reasoning: parsed.reasoning || '',
      confidence: parsed.confidence || 50
    };
  } catch (error) {
    console.error('Error calling backend API:', error);
    console.log('Falling back to rule-based AI...');
    return fallbackDecision();
  }
};

export default {
  getAIDecision,
};

