import { useState } from 'react';
import Card from './Card';
import Player from './Player';
import './PokerTable.css';

const PokerTable = ({ players, communityCards, pot, currentPlayerIndex }) => {
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
            {communityCards.length > 0 ? (
              communityCards.map((card, index) => (
                <Card 
                  key={index} 
                  rank={card.rank} 
                  suit={card.suit} 
                  faceDown={card.faceDown}
                />
              ))
            ) : (
              <div style={{ color: '#888', fontSize: '14px', padding: '20px' }}>
                Preflop - No community cards yet
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
              isDealer={player.isDealer}
              isFolded={player.isFolded}
            />
          ))}
        </div>
      </div>
    </div>
  );
};

export default PokerTable;

