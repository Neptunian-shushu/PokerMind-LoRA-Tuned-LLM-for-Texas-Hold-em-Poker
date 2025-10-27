import './Card.css';

const Card = ({ rank, suit, faceDown = false }) => {
  const suitSymbols = {
    hearts: '♥',
    diamonds: '♦',
    clubs: '♣',
    spades: '♠'
  };

  const suitColors = {
    hearts: 'red',
    diamonds: 'red',
    clubs: 'black',
    spades: 'black'
  };

  if (faceDown) {
    return (
      <div className="card card-back">
        <div className="card-pattern"></div>
      </div>
    );
  }

  return (
    <div className={`card card-face ${suitColors[suit]}`}>
      <div className="card-corner top-left">
        <div className="rank">{rank}</div>
        <div className="suit">{suitSymbols[suit]}</div>
      </div>
      <div className="card-center">
        <span className="suit-large">{suitSymbols[suit]}</span>
      </div>
      <div className="card-corner bottom-right">
        <div className="rank">{rank}</div>
        <div className="suit">{suitSymbols[suit]}</div>
      </div>
    </div>
  );
};

export default Card;

