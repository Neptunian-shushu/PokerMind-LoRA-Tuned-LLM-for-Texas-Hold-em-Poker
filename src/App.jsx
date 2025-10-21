import { useState, useEffect, useCallback } from 'react';
import PokerTable from './components/PokerTable';
import ControlPanel from './components/ControlPanel';
import { initializeGame, playerAction, moveToNextPlayer, isRoundComplete, getAvailableActions, determineWinner } from './utils/gameLogic';
import { getAIDecision } from './services/aiService';
import './App.css';

function App() {
  const [gameState, setGameState] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [message, setMessage] = useState('');
  const [gameOver, setGameOver] = useState(false);
  const [gameStarted, setGameStarted] = useState(false);

  // initialize game
  const startNewGame = useCallback(() => {
    const newGame = initializeGame(['You', 'AI Bot 1', 'AI Bot 2', 'AI Bot 3', 'AI Bot 4', 'AI Bot 5'], 1000, 10, 20);
    setGameState(newGame);
    setGameOver(false);
    setGameStarted(true);
    setMessage('Game started! Waiting for action...');
  }, []);

  // process AI player's turn
  const processAITurn = useCallback(async (state) => {
    if (!state || gameOver) return state;

    const currentPlayer = state.players[state.currentPlayerIndex];
    
    // if current player is human, skip
    if (currentPlayer.isHuman) {
      return state;
    }

    // if current player has folded, skip
    if (currentPlayer.isFolded) {
      return moveToNextPlayer(state);
    }

    setIsProcessing(true);
    setMessage(`${currentPlayer.name} is thinking...`);

    await new Promise(resolve => setTimeout(resolve, 1000));

    try {
      // get AI decision
      const decision = await getAIDecision(state, state.currentPlayerIndex);
      
      setMessage(`${currentPlayer.name} decides to ${decision.action.toUpperCase()}${decision.raiseAmount > 0 ? ' $' + decision.raiseAmount : ''}`);
      
      // execute AI action
      let newState = playerAction(state, decision.action, decision.raiseAmount);
      
      // check if round is complete
      if (isRoundComplete(newState)) {
        // preflop round complete
        const result = determineWinner(newState);
        
        if (result.isFinished) {
          // game over in preflop (all players folded)
          const winningPlayer = newState.players[result.winnerId];
          newState.players[result.winnerId].chips += newState.pot;
          setMessage(`${winningPlayer.name} wins the pot of $${newState.pot}!`);
          setGameOver(true);
          setIsProcessing(false);
          return newState;
        } else {
          // TODO:  postflop logic here
          console.log('=== Preflop Complete - Ready for Postflop ===');
          console.log('Active Players:', result.activePlayers.length);
          console.log('Pot:', newState.pot);
          
          // temporary: show message and end game
          setMessage(`Preflop complete! ${result.activePlayers.length} players remain. (Postflop TODO)`);
          setGameOver(true);
          setIsProcessing(false);
          return newState;
        }
      }
      
      // move to next player
      newState = moveToNextPlayer(newState);
      setIsProcessing(false);
      
      return newState;
      
    } catch (error) {
      console.error('Error processing AI turn:', error);
      setMessage('Error processing AI action');
      setIsProcessing(false);
      return state;
    }
  }, [gameOver]);

  // when game state changes and it's AI's turn, automatically process AI turn
  useEffect(() => {
    if (!gameState || isProcessing || gameOver) return;

    const currentPlayer = gameState.players[gameState.currentPlayerIndex];
    
    if (!currentPlayer.isHuman && !currentPlayer.isFolded) {
      // use setTimeout to ensure state is updated before processing
      const timer = setTimeout(async () => {
        const newState = await processAITurn(gameState);
        if (newState !== gameState) {
          setGameState(newState);
        }
      }, 100);
      
      return () => clearTimeout(timer);
    }
  }, [gameState, isProcessing, gameOver, processAITurn]);

  // process player action
  const handlePlayerAction = async (action, raiseAmount = 0) => {
    if (!gameState || isProcessing || gameOver) return;

    const currentPlayer = gameState.players[gameState.currentPlayerIndex];
    
    // ensure it's human player's turn
    if (!currentPlayer.isHuman) {
      setMessage('Not your turn!');
      return;
    }

    setIsProcessing(true);

    // execute player action
    let newState = playerAction(gameState, action, raiseAmount);
    
    setMessage(`You ${action.toUpperCase()}${raiseAmount > 0 ? ' $' + raiseAmount : ''}`);

    // check if round is complete
    if (isRoundComplete(newState)) {
      const result = determineWinner(newState);
      
      if (result.isFinished) {
        // game over in preflop (all players folded)
        const winningPlayer = newState.players[result.winnerId];
        newState.players[result.winnerId].chips += newState.pot;
        setMessage(`${winningPlayer.name} wins the pot of $${newState.pot}!`);
        setGameOver(true);
        setGameState(newState);
        setIsProcessing(false);
        return;
      } else {
        // need to continue to postflop
        // TODO: postflop logic here
        console.log('=== Preflop Complete - Ready for Postflop ===');
        console.log('Active Players:', result.activePlayers.length);
        console.log('Pot:', newState.pot);
        
        // temporary: show message and end game
        setMessage(`Preflop complete! ${result.activePlayers.length} players remain. (Postflop TODO)`);
        setGameOver(true);
        setGameState(newState);
        setIsProcessing(false);
        return;
      }
    }

    // move to next player
    newState = moveToNextPlayer(newState);
    setGameState(newState);
    setIsProcessing(false);
  };

  const currentPlayer = gameState?.players[gameState.currentPlayerIndex];
  const isPlayerTurn = currentPlayer && currentPlayer.isHuman;
  const availableActions = isPlayerTurn ? getAvailableActions(gameState) : null;

  return (
    <div className="app">
      <header className="app-header">
        <h1 className="app-title">
          ♠️ PokerMind AI ♥️
        </h1>
        <p className="app-subtitle">
          Texas Hold'em - You vs 5 AI Players
        </p>
      </header>

      <main className="app-main">
        {!gameStarted ? (
          // game not started, show start button
          <div className="start-game-overlay">
            <div className="start-game-container">
              <div className="start-game-content">
                <h2 className="start-game-title">Welcome to PokerMind</h2>
                <p className="start-game-description">
                  Play Texas Hold'em Poker against 5 AI opponents powered by AI
                </p>
                <div className="game-settings">
                  <div className="setting-item">
                    <span className="setting-icon">👥</span>
                    <span>6 Players (You + 5 AI)</span>
                  </div>
                  <div className="setting-item">
                    <span className="setting-icon">💰</span>
                    <span>Starting Chips: $1,000</span>
                  </div>
                  <div className="setting-item">
                    <span className="setting-icon">🎰</span>
                    <span>Blinds: $10 / $20</span>
                  </div>
                </div>
                <button 
                  className="start-game-button"
                  onClick={startNewGame}
                >
                  <span className="button-icon">🎮</span>
                  Start Game
                </button>
              </div>
            </div>
          </div>
        ) : (
          // game started, show normal interface
          <>
            <div className="poker-section">
              <PokerTable 
                players={gameState.players}
                communityCards={gameState.communityCards}
                pot={gameState.pot}
                currentPlayerIndex={gameState.currentPlayerIndex}
              />
            </div>

            <div className="control-section">
              <ControlPanel 
                onAction={handlePlayerAction}
                isPlayerTurn={isPlayerTurn}
                isProcessing={isProcessing}
                availableActions={availableActions}
                message={message}
                gameOver={gameOver}
                onNewGame={startNewGame}
                gameState={gameState}
              />
            </div>
          </>
        )}
      </main>

      <footer className="app-footer">
        <p>Built with React + Vite | Powered by DeepSeek AI</p>
      </footer>
    </div>
  );
}

export default App;
