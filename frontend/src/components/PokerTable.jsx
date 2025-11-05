import Card from './Card';
import Player from './Player';
import './PokerTable.css';
import { BETTING_ROUNDS } from '../utils/gameLogic';

const PokerTable = ({ 
  players, 
  communityCards, 
  pot, 
  currentPlayerIndex, 
  phase = BETTING_ROUNDS.PREFLOP,
  lastWinnerId = null
}) => {
  return (
    <div className="poker-table-container">
      <div className="poker-table">
        {/* Community Cards Area */}
        <div className="community-cards">
          <div className="pot-info">
            <div className="pot-label">POT</div>
            <div className="pot-amount">
              <span className="chip-icon">🎰</span>
              ${pot.toLocaleString()}
            </div>
          </div>
          <div className="cards-area">
            <div className="phase-indicator">
              {phase.toUpperCase()}
            </div>
            {communityCards.length > 0 ? (
              <div className="community-cards-grid">
                {communityCards.map((card, index) => (
                  <Card 
                    key={index} 
                    rank={card.rank} 
                    suit={card.suit} 
                    faceDown={card.faceDown}
                    className={`community-card ${
                      index < 3 ? 'flop' : 
                      index === 3 ? 'turn' : 'river'
                    }`}
                  />
                ))}
              </div>
            ) : (
              <div className="no-cards-message">
                Preflop - Waiting for community cards
              </div>
            )}
          </div>
        </div>

        {/* Players positioned around the table */}
        <div className="players-grid">
          {players.map((player, index) => (
            <Player
              key={index}
              name={player.name}
              chips={player.chips}
              cards={player.cards}
              isActive={index === currentPlayerIndex}
              position={player.position}
              bet={player.bet}
              action={player.action}
              lastAction={player.lastAction}
              isDealer={player.isDealer}
              isFolded={player.isFolded}
              isWinner={lastWinnerId !== null && player.id === lastWinnerId}
            />
          ))}
        </div>
      </div>
    </div>
  );
};

export default PokerTable;

