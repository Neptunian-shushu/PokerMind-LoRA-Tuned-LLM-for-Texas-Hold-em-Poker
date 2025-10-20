import { useState } from 'react';
import Card from './Card';
import Player from './Player';
import './PokerTable.css';

const PokerTable = ({ players, communityCards, pot }) => {
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
              <>
                <Card faceDown={true} />
                <Card faceDown={true} />
                <Card faceDown={true} />
                <Card faceDown={true} />
                <Card faceDown={true} />
              </>
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
              isActive={player.isActive}
              position={player.position}
              bet={player.bet}
              action={player.action}
              isDealer={player.isDealer}
            />
          ))}
        </div>
      </div>
    </div>
  );
};

export default PokerTable;

