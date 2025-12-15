import PropTypes from 'prop-types';

const ForecastTable = ({ forecast, loading, error }) => {
  if (loading) {
    return (
      <div className="card">
        <h3>Forecast Results</h3>
        <div style={{ textAlign: 'center', padding: '2rem' }}>
          <span className="loading"></span>
          <div style={{ marginTop: '1rem' }}>Loading forecast...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card">
        <h3>Forecast Results</h3>
        <div className="error">{error}</div>
      </div>
    );
  }

  if (!forecast || forecast.length === 0) {
    return (
      <div className="card">
        <h3>Forecast Results</h3>
        <div>No forecast data available</div>
      </div>
    );
  }

  return (
    <div className="card">
      <h3>Price Forecast</h3>
      <table className="table">
        <thead>
          <tr>
            <th>Date</th>
            <th>Predicted Price</th>
            <th>Lower Bound</th>
            <th>Upper Bound</th>
          </tr>
        </thead>
        <tbody>
          {forecast.map((row, index) => (
            <tr key={index}>
              <td>{new Date(row.ds).toLocaleDateString()}</td>
              <td>${row.yhat.toFixed(2)}</td>
              <td>{row.yhat_lower ? `$${row.yhat_lower.toFixed(2)}` : '-'}</td>
              <td>{row.yhat_upper ? `$${row.yhat_upper.toFixed(2)}` : '-'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

ForecastTable.propTypes = {
  forecast: PropTypes.arrayOf(PropTypes.shape({
    ds: PropTypes.string.isRequired,
    yhat: PropTypes.number.isRequired,
    yhat_lower: PropTypes.number,
    yhat_upper: PropTypes.number
  })),
  loading: PropTypes.bool,
  error: PropTypes.string
};

export default ForecastTable;
