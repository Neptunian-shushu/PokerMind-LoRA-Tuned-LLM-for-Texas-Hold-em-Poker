import { useState } from 'react';
import './ControlPanel.css';

const ControlPanel = ({ onSubmit, decision, isLoading }) => {
  const [condition, setCondition] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (condition.trim()) {
      onSubmit(condition);
    }
  };

  const exampleConditions = [
    "Position: BTN, Hand: AK, Pot: 100, Stack: 1000",
    "Position: SB, Hand: QQ, Pot: 200, Stack: 800, Action: Facing raise",
    "Flop: A♠ K♥ 7♦, Hand: A♣ Q♦, Pot: 300, Position: BB"
  ];

  const handleExampleClick = (example) => {
    setCondition(example);
  };

  return (
    <div className="control-panel">
      <div className="control-section">
        <h2 className="section-title">📝 Game Condition</h2>
        <form onSubmit={handleSubmit} className="input-form">
          <textarea
            value={condition}
            onChange={(e) => setCondition(e.target.value)}
            placeholder="Enter game condition (position, hand, pot size, stack size, etc.)&#10;&#10;Example:&#10;Position: BTN&#10;Hand: AK suited&#10;Pot: 150&#10;Stack: 1000&#10;Board: A♠ K♥ 7♦"
            className="condition-input"
            rows="6"
          />
          <div className="example-chips">
            <span className="example-label">Quick examples:</span>
            {exampleConditions.map((example, index) => (
              <button
                key={index}
                type="button"
                onClick={() => handleExampleClick(example)}
                className="example-chip"
              >
                {example}
              </button>
            ))}
          </div>
          <button 
            type="submit" 
            className="submit-button"
            disabled={isLoading || !condition.trim()}
          >
            {isLoading ? (
              <>
                <span className="spinner"></span>
                Processing...
              </>
            ) : (
              <>
                🎯 Get Decision
              </>
            )}
          </button>
        </form>
      </div>

      <div className="control-section">
        <h2 className="section-title">🤖 AI Decision</h2>
        <div className="decision-output">
          {isLoading ? (
            <div className="loading-state">
              <div className="spinner-large"></div>
              <p>Analyzing game state...</p>
            </div>
          ) : decision ? (
            <div className="decision-content">
              <div className="decision-action">
                <span className="action-label">Recommended Action:</span>
                <span className="action-value">{decision.action || decision}</span>
              </div>
              {decision.reasoning && (
                <div className="decision-reasoning">
                  <span className="reasoning-label">Reasoning:</span>
                  <p className="reasoning-text">{decision.reasoning}</p>
                </div>
              )}
              {decision.confidence && (
                <div className="confidence-bar-container">
                  <span className="confidence-label">Confidence:</span>
                  <div className="confidence-bar">
                    <div 
                      className="confidence-fill" 
                      style={{ width: `${decision.confidence}%` }}
                    >
                      {decision.confidence}%
                    </div>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="empty-state">
              <p>💭 Enter a game condition above and click "Get Decision" to see the AI's recommendation.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ControlPanel;

