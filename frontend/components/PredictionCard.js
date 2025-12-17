import PropTypes from 'prop-types';

const PredictionCard = ({ model, prediction, probability, confidence, error }) => {
  if (error) {
    return (
      <div className="card">
        <div className="model-name">{model}</div>
        <div className="error">Error: {error}</div>
      </div>
    );
  }

  if (!prediction && prediction !== 0) {
    return (
      <div className="card">
        <div className="model-name">{model}</div>
        <div className="prediction-value">No prediction available</div>
      </div>
    );
  }

  const isVolatilityModel = typeof model === 'string' && model.toLowerCase().includes('volatility');
  const numericPrediction =
    typeof prediction === 'number'
      ? prediction
      : typeof prediction === 'string'
      ? Number(prediction)
      : NaN;

  const displayPrediction = (() => {
    if (isVolatilityModel && Number.isFinite(numericPrediction)) {
      return `${(numericPrediction * 100).toFixed(2)}%/day`;
    }
    return prediction;
  })();

  return (
    <div className="card">
      <div className="model-name">{model}</div>
      <div className="prediction-value">{displayPrediction}</div>
      {isVolatilityModel && Number.isFinite(numericPrediction) && (
        <div style={{ fontSize: '0.875rem', color: '#6b7280', marginTop: '0.25rem' }}>
          Unit: σ(log returns) (dimensionless) — shown as approx % per day
        </div>
      )}
      {probability && (
        <div style={{ fontSize: '0.875rem', color: '#6b7280', marginTop: '0.25rem' }}>
          Probability: {(probability * 100).toFixed(1)}%
        </div>
      )}
      {confidence && (
        <div style={{ fontSize: '0.875rem', color: '#6b7280', marginTop: '0.25rem' }}>
          Confidence: {(confidence * 100).toFixed(1)}%
        </div>
      )}
    </div>
  );
};

PredictionCard.propTypes = {
  model: PropTypes.string.isRequired,
  prediction: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
  probability: PropTypes.number,
  confidence: PropTypes.number,
  error: PropTypes.string
};

export default PredictionCard;
