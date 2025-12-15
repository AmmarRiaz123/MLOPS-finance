import PropTypes from 'prop-types';

const RegimeIndicator = ({ regime, score, loading, error, model }) => {
  if (loading) {
    return (
      <div className="card">
        <h3>Market Regime</h3>
        <div style={{ textAlign: 'center', padding: '2rem' }}>
          <span className="loading"></span>
          <div style={{ marginTop: '1rem' }}>Analyzing regime...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card">
        <h3>Market Regime</h3>
        <div className="error">{error}</div>
      </div>
    );
  }

  if (!regime) {
    return (
      <div className="card">
        <h3>Market Regime</h3>
        <div>No regime data available</div>
      </div>
    );
  }

  const getRegimeClass = (regime) => {
    switch (regime.toLowerCase()) {
      case 'bull': return 'regime-bull';
      case 'bear': return 'regime-bear';
      default: return 'regime-neutral';
    }
  };

  const getRegimeEmoji = (regime) => {
    switch (regime.toLowerCase()) {
      case 'bull': return '📈';
      case 'bear': return '📉';
      default: return '➡️';
    }
  };

  return (
    <div className="card">
      <h3>Market Regime</h3>
      {model && <div className="model-name">{model}</div>}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '1rem' }}>
        <span style={{ fontSize: '2rem' }}>{getRegimeEmoji(regime)}</span>
        <div>
          <div className={`prediction-value ${getRegimeClass(regime)}`}>
            {regime.charAt(0).toUpperCase() + regime.slice(1)} Market
          </div>
          {score !== undefined && (
            <div style={{ fontSize: '0.875rem', color: '#6b7280', marginTop: '0.25rem' }}>
              Confidence: {(score * 100).toFixed(1)}%
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

RegimeIndicator.propTypes = {
  regime: PropTypes.string,
  score: PropTypes.number,
  loading: PropTypes.bool,
  error: PropTypes.string,
  model: PropTypes.string
};

export default RegimeIndicator;
