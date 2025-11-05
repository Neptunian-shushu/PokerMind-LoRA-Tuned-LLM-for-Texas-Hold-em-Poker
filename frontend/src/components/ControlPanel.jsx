import { useState } from 'react';
import './ControlPanel.css';

const ControlPanel = ({ 
  onAction, 
  isPlayerTurn, 
  isProcessing, 
  availableActions,
  message,
  gameOver,
  onNewGame,
  gameState
}) => {
  const [raiseAmount, setRaiseAmount] = useState('');

  const handleAction = (action) => {
    if (action === 'raise') {
      const amount = parseFloat(raiseAmount);
      if (isNaN(amount) || amount < availableActions.minRaise) {
        alert(`Minimum raise is $${availableActions.minRaise}`);
        return;
      }
      if (amount > availableActions.maxRaise) {
        alert(`Maximum raise is $${availableActions.maxRaise}`);
        return;
      }
      onAction(action, amount);
      setRaiseAmount('');
    } else {
      onAction(action, 0);
    }
  };

  const currentPlayer = gameState?.players[gameState.currentPlayerIndex];

  return (
    <div className="control-panel">
      {/* Game Status */}
      <div className="control-section">
        <h2 className="section-title">📊 Game Status</h2>
        <div className="status-display">
          <div className="status-message">
            {message || 'Waiting...'}
          </div>
          
          {currentPlayer && (
            <div className="current-turn">
              <strong>Current Turn:</strong> {currentPlayer.name}
              {isPlayerTurn && <span className="your-turn-badge">YOUR TURN!</span>}
            </div>
          )}

          <div className="game-info">
            <div className="info-item">
              <span className="info-label">Phase:</span>
              <span className="info-value">{gameState?.phase?.toUpperCase() || 'PREFLOP'}</span>
            </div>
            <div className="info-item">
              <span className="info-label">Pot:</span>
              <span className="info-value">${gameState?.pot || 0}</span>
            </div>
            <div className="info-item">
              <span className="info-label">Current Bet:</span>
              <span className="info-value">${gameState?.currentBet || 0}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Player Actions */}
      <div className="control-section">
        <h2 className="section-title">🎮 Your Actions</h2>
        
        {gameOver ? (
          <div className="game-over-section">
            <div className="game-over-message">
              <h3>🎉 Game Complete!</h3>
              {message.split(' | ').map((line, index) => (
                <p key={index} className={`game-result-line ${
                  line.includes('WINS!') ? 'winner-line' : 
                  line.includes('Net Gain') ? 'gain-line' : ''
                }`}>
                  {line}
                </p>
              ))}
            </div>
            <button 
              onClick={onNewGame}
              className="new-game-button"
            >
              🎲 Start New Game
            </button>
          </div>
        ) : isPlayerTurn && availableActions ? (
          <div className="actions-container">
            <div className="action-info">
              {availableActions.toCall > 0 && (
                <p className="call-info">
                  Amount to call: <strong>${availableActions.toCall}</strong>
                </p>
              )}
            </div>

            <div className="action-buttons">
              {availableActions.actions.includes('fold') && (
                <button 
                  onClick={() => handleAction('fold')}
                  className="action-btn fold-btn"
                  disabled={isProcessing}
                >
                  🚫 Fold
                </button>
              )}

              {availableActions.actions.includes('check') && (
                <button 
                  onClick={() => handleAction('check')}
                  className="action-btn check-btn"
                  disabled={isProcessing}
                >
                  ✓ Check
                </button>
              )}

              {availableActions.actions.includes('call') && (
                <button 
                  onClick={() => handleAction('call')}
                  className="action-btn call-btn"
                  disabled={isProcessing}
                >
                  📞 Call ${availableActions.toCall}
                </button>
              )}

              {availableActions.actions.includes('raise') && (
                <div className="raise-section">
                  <input 
                    type="number"
                    value={raiseAmount}
                    onChange={(e) => setRaiseAmount(e.target.value)}
                    placeholder={`Min: ${availableActions.minRaise}`}
                    min={availableActions.minRaise}
                    max={availableActions.maxRaise}
                    className="raise-input"
                    disabled={isProcessing}
                  />
                  <button 
                    onClick={() => handleAction('raise')}
                    className="action-btn raise-btn"
                    disabled={isProcessing || !raiseAmount}
                  >
                    💰 Raise
                  </button>
                </div>
              )}

              {availableActions.actions.includes('all-in') && (
                <button 
                  onClick={() => handleAction('all-in')}
                  className="action-btn allin-btn"
                  disabled={isProcessing}
                >
                  🔥 All-In
                </button>
              )}
            </div>

            {availableActions.actions.includes('raise') && (
              <div className="raise-range">
                <small>
                  Raise range: ${availableActions.minRaise} - ${availableActions.maxRaise}
                </small>
              </div>
            )}
          </div>
        ) : (
          <div className="waiting-section">
            {isProcessing ? (
              <>
                <div className="spinner-large"></div>
                <p>Processing action...</p>
              </>
            ) : (
              <p>Waiting for other players...</p>
            )}
          </div>
        )}
      </div>

      {/* Player Info */}
      {gameState?.players[0] && (
        <div className="control-section">
          <h2 className="section-title">👤 Your Info</h2>
          <div className="player-stats">
            <div className="stat-item">
              <span className="stat-label">Chips:</span>
              <span className="stat-value">${gameState.players[0].chips}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Bet This Round:</span>
              <span className="stat-value">${gameState.players[0].totalBetThisRound}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Hand:</span>
              <span className="stat-value">
                {gameState.players[0].cards.map(card => 
                  `${card.rank}${getSuitSymbol(card.suit)}`
                ).join(' ')}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Helper function
const getSuitSymbol = (suit) => {
  const symbols = {
    hearts: '♥',
    diamonds: '♦',
    clubs: '♣',
    spades: '♠'
  };
  return symbols[suit] || '';
};

export default ControlPanel;
