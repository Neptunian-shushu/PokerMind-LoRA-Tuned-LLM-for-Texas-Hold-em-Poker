import { useState } from 'react';
import PokerTable from './components/PokerTable';
import ControlPanel from './components/ControlPanel';
import './App.css';

function App() {
  const [decision, setDecision] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  // Sample game state
  const [gameState] = useState({
    players: [
      {
        name: 'You (Hero)',
        chips: 1500,
        cards: [
          { rank: 'A', suit: 'spades', faceDown: false },
          { rank: 'K', suit: 'hearts', faceDown: false }
        ],
        isActive: true,
        position: 'bottom',
        bet: 50,
        action: '',
        isDealer: false
      },
      {
        name: 'Player 2',
        chips: 2000,
        cards: [
          { rank: '', suit: '', faceDown: true },
          { rank: '', suit: '', faceDown: true }
        ],
        isActive: false,
        position: 'left',
        bet: 50,
        action: 'Call',
        isDealer: true
      },
      {
        name: 'Player 3',
        chips: 1200,
        cards: [
          { rank: '', suit: '', faceDown: true },
          { rank: '', suit: '', faceDown: true }
        ],
        isActive: false,
        position: 'top',
        bet: 0,
        action: 'Fold',
        isDealer: false
      },
      {
        name: 'Player 4',
        chips: 1800,
        cards: [
          { rank: '', suit: '', faceDown: true },
          { rank: '', suit: '', faceDown: true }
        ],
        isActive: false,
        position: 'right',
        bet: 100,
        action: 'Raise',
        isDealer: false
      }
    ],
    communityCards: [
      { rank: 'A', suit: 'diamonds', faceDown: false },
      { rank: 'K', suit: 'clubs', faceDown: false },
      { rank: '7', suit: 'hearts', faceDown: false },
      { rank: '', suit: '', faceDown: true },
      { rank: '', suit: '', faceDown: true }
    ],
    pot: 300
  });

  const handleSubmitCondition = async (condition) => {
    setIsLoading(true);
    setDecision(null);

    // Simulate API call to your LoRA model
    // Replace this with your actual API endpoint
    try {
      // Simulated delay and response
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      // Mock response - replace with actual API call
      const mockDecision = {
        action: 'RAISE',
        reasoning: 'With top two pair (Aces and Kings) and strong kicker, raising is optimal to build the pot and protect against draws. Your hand is likely ahead of most opponent ranges in this position.',
        confidence: 85
      };

      setDecision(mockDecision);
    } catch (error) {
      console.error('Error getting decision:', error);
      setDecision({
        action: 'ERROR',
        reasoning: 'Failed to get decision from AI model. Please try again.',
        confidence: 0
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1 className="app-title">
          ♠️ PokerMind AI ♥️
        </h1>
        <p className="app-subtitle">
          LoRA-Tuned LLM for Texas Hold'em Decision Making
        </p>
      </header>

      <main className="app-main">
        <div className="poker-section">
          <PokerTable 
            players={gameState.players}
            communityCards={gameState.communityCards}
            pot={gameState.pot}
          />
        </div>

        <div className="control-section">
          <ControlPanel 
            onSubmit={handleSubmitCondition}
            decision={decision}
            isLoading={isLoading}
          />
        </div>
      </main>

      <footer className="app-footer">
        <p>Built with React + Vite | Powered by LoRA Fine-tuned LLM</p>
      </footer>
    </div>
  );
}

export default App;
