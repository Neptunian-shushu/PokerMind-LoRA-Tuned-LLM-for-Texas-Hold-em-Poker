import Card from './Card';
import './Player.css';

const Player = ({ 
  name, 
  chips, 
  cards = [], 
  isActive = false, 
  position = 'bottom',
  bet = 0,
  action = '',
  isDealer = false,
  isFolded = false
}) => {
  return (
    <div className={`player player-${position} ${isActive ? 'active' : ''} ${isFolded ? 'folded' : ''}`}>
      {isDealer && <div className="dealer-button">D</div>}
      
      <div className="player-info">
        <div className="player-name">{name}</div>
        <div className="player-chips">
          <span className="chip-icon">🎰</span>
          ${chips.toLocaleString()}
        </div>
      </div>

      <div className="player-cards">
        {cards.map((card, index) => (
          <Card 
            key={index} 
            rank={card.rank} 
            suit={card.suit} 
            faceDown={card.faceDown}
          />
        ))}
      </div>

      {bet > 0 && (
        <div className="player-bet">
          <span className="chip-icon">🎰</span>
          ${bet.toLocaleString()}
        </div>
      )}

      {action && (
        <div className="player-action">{action}</div>
      )}
    </div>
  );
};

export default Player;

