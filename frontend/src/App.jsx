import { useState, useEffect, useCallback } from 'react';
import PokerTable from './components/PokerTable';
import ControlPanel from './components/ControlPanel';
import { initializeGame, playerAction, moveToNextPlayer, isRoundComplete, getAvailableActions, determineWinner, BETTING_ROUNDS, advanceToNextRound } from './utils/gameLogic';
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
        const result = determineWinner(newState);
        
        if (result.isFinished) {
          // 游戏结束（所有玩家弃牌或摊牌）
          const updatedState = result.gameState;
          const winningPlayer = updatedState.players[result.winnerId];

          // 仅显示三条信息：赢家、赢得金额、你的最终筹码（含收益）
          const winnerStats = result.playerStats?.find(p => p.id === result.winnerId);
          const amountWon = winnerStats?.netGain ?? updatedState.pot;
          const yourBaseChips = updatedState.players[0]?.chips ?? 0;
          const yourFinalChips = result.winnerId === 0 ? yourBaseChips + updatedState.pot : yourBaseChips;

          const resultMessage = [
            `Winner: ${winningPlayer.name}`,
            `Won: $${amountWon}`,
            `Your Chips: $${yourFinalChips}`
          ];

          setMessage(resultMessage.join(' | '));
          setGameOver(true);
          setIsProcessing(false);
          return result.gameState;
        } else if (newState.phase === BETTING_ROUNDS.SHOWDOWN) {
          // 到达摊牌阶段 - 处理摊牌结果
          const result = determineWinner(newState);
          const updatedState = result.gameState;
          const winningPlayer = updatedState.players[result.winnerId];

          // 仅显示三条信息：赢家、赢得金额、你的最终筹码（含收益）
          const winnerStats = result.playerStats?.find(p => p.id === result.winnerId);
          const amountWon = winnerStats?.netGain ?? updatedState.pot;
          const yourBaseChips = updatedState.players[0]?.chips ?? 0;
          const yourFinalChips = result.winnerId === 0 ? yourBaseChips + updatedState.pot : yourBaseChips;

          const resultMessage = [
            `Winner: ${winningPlayer.name}`,
            `Won: $${amountWon}`,
            `Your Chips: $${yourFinalChips}`
          ];

          setMessage(resultMessage.join(' | '));
          setGameOver(true);
          setIsProcessing(false);
          return result.gameState;
        } else {
          // 进入下一个回合
          console.log(`=== ${newState.phase} Complete - Moving to Next Round ===`);
          newState = advanceToNextRound(newState);

          // 如果推进后是摊牌，立即结算，不再等待额外一次"Check"
          if (newState.phase === BETTING_ROUNDS.SHOWDOWN) {
            const sdResult = determineWinner(newState);
            const updatedState = sdResult.gameState;
            const winningPlayer = updatedState.players[sdResult.winnerId];

            const winnerStats = sdResult.playerStats?.find(p => p.id === sdResult.winnerId);
            const amountWon = winnerStats?.netGain ?? updatedState.pot;
            const yourBaseChips = updatedState.players[0]?.chips ?? 0;
            const yourFinalChips = sdResult.winnerId === 0 ? yourBaseChips + updatedState.pot : yourBaseChips;

            const resultMessage = [
              `Winner: ${winningPlayer.name}`,
              `Won: $${amountWon}`,
              `Your Chips: $${yourFinalChips}`
            ];

            setMessage(resultMessage.join(' | '));
            setGameOver(true);
            setIsProcessing(false);
            return sdResult.gameState;
          }

          setMessage(`Moving to ${newState.phase.toUpperCase()}`);
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
    
    // 更新游戏状态显示
    const updateGameStatus = () => {
      const activePlayers = gameState.players.filter(p => !p.isFolded);
      let statusMessage = '';

      switch(gameState.phase) {
        case BETTING_ROUNDS.PREFLOP:
          statusMessage = `Preflop - ${activePlayers.length} players, pot: $${gameState.pot}`;
          break;
        case BETTING_ROUNDS.FLOP:
          statusMessage = `Flop - ${gameState.communityCards.slice(0, 3).map(c => `${c.rank}${c.suit}`).join(' ')}, pot: $${gameState.pot}`;
          break;
        case BETTING_ROUNDS.TURN:
          statusMessage = `Turn - ${gameState.communityCards[3].rank}${gameState.communityCards[3].suit}, pot: $${gameState.pot}`;
          break;
        case BETTING_ROUNDS.RIVER:
          statusMessage = `River - ${gameState.communityCards[4].rank}${gameState.communityCards[4].suit}, pot: $${gameState.pot}`;
          break;
        case BETTING_ROUNDS.SHOWDOWN:
          statusMessage = `Showdown - Final pot: $${gameState.pot}`;
          break;
      }

      setMessage(statusMessage);
    };

    updateGameStatus();
    
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
        // 游戏结束（包括弃牌和摊牌的情况）
        const updatedState = result.gameState;
        const winningPlayer = updatedState.players[result.winnerId];

        const winnerStats = result.playerStats?.find(p => p.id === result.winnerId);
        const amountWon = winnerStats?.netGain ?? updatedState.pot;
        const yourBaseChips = updatedState.players[0]?.chips ?? 0;
        const yourFinalChips = result.winnerId === 0 ? yourBaseChips + updatedState.pot : yourBaseChips;

        const resultMessage = [
          `Winner: ${winningPlayer.name}`,
          `Won: $${amountWon}`,
          `Your Chips: $${yourFinalChips}`
        ];

        setMessage(resultMessage.join(' | '));
        setGameOver(true);
        setGameState(updatedState);
        setIsProcessing(false);
        return;
      } else if (newState.phase === BETTING_ROUNDS.SHOWDOWN) {
        // 到达摊牌阶段 - 处理摊牌结果（仅显示三条信息）
        const result = determineWinner(newState);
        const updatedState = result.gameState;
        const winningPlayer = updatedState.players[result.winnerId];

        const winnerStats = result.playerStats?.find(p => p.id === result.winnerId);
        const amountWon = winnerStats?.netGain ?? updatedState.pot;
        const yourBaseChips = updatedState.players[0]?.chips ?? 0;
        const yourFinalChips = result.winnerId === 0 ? yourBaseChips + updatedState.pot : yourBaseChips;

        const resultMessage = [
          `Winner: ${winningPlayer.name}`,
          `Won: $${amountWon}`,
          `Your Chips: $${yourFinalChips}`
        ];

        setMessage(resultMessage.join(' | '));
        setGameOver(true);
        setGameState(updatedState);
        setIsProcessing(false);
        return;
      } else {
        // 进入下一个回合
        console.log(`=== ${newState.phase} Complete - Moving to Next Round ===`);
        newState = advanceToNextRound(newState);

        // 如果推进后是摊牌，立即结算，不再等待额外一次"Check"
        if (newState.phase === BETTING_ROUNDS.SHOWDOWN) {
          const sdResult = determineWinner(newState);
          const updatedState = sdResult.gameState;
          const winningPlayer = updatedState.players[sdResult.winnerId];

          const winnerStats = sdResult.playerStats?.find(p => p.id === sdResult.winnerId);
          const amountWon = winnerStats?.netGain ?? updatedState.pot;
          const yourBaseChips = updatedState.players[0]?.chips ?? 0;
          const yourFinalChips = sdResult.winnerId === 0 ? yourBaseChips + updatedState.pot : yourBaseChips;

          const resultMessage = [
            `Winner: ${winningPlayer.name}`,
            `Won: $${amountWon}`,
            `Your Chips: $${yourFinalChips}`
          ];

          setMessage(resultMessage.join(' | '));
          setGameOver(true);
          setGameState(updatedState);
          setIsProcessing(false);
          return;
        }

        setMessage(`Moving to ${newState.phase.toUpperCase()}`);
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
                phase={gameState.phase}
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
