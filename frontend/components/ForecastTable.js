import { useState } from 'react';
import PropTypes from 'prop-types';


const HistoricalDataForm = ({ onSubmit, loading }) => {
  const [periods, setPeriods] = useState(5);
  const [historyCount, setHistoryCount] = useState(5);
  const [history, setHistory] = useState([
    { date: '2025-12-10', open: 150.25, high: 152.00, low: 149.50, close: 151.20, volume: 1200000 },
    { date: '2025-12-11', open: 151.30, high: 153.10, low: 150.80, close: 152.75, volume: 1350000 },
    { date: '2025-12-12', open: 152.80, high: 154.00, low: 152.00, close: 153.50, volume: 1400000 },
    { date: '2025-12-13', open: 153.60, high: 155.20, low: 153.00, close: 154.80, volume: 1250000 },
    { date: '2025-12-14', open: 154.90, high: 156.00, low: 154.50, close: 155.75, volume: 1300000 }
  ]);

  const updateHistoryCount = (count) => {
    setHistoryCount(count);
    const newHistory = [...history];
    
    if (count > history.length) {
      for (let i = history.length; i < count; i++) {
        const prevDate = newHistory[i - 1]?.date || '2025-12-10';
        const nextDate = new Date(prevDate);
        nextDate.setDate(nextDate.getDate() + 1);
        
        newHistory.push({
          date: nextDate.toISOString().split('T')[0],
          open: 150.00,
          high: 152.00,
          low: 149.00,
          close: 151.00,
          volume: 1000000
        });
      }
    } else {
      newHistory.splice(count);
    }
    
    setHistory(newHistory);
  };

  const updateHistoryRow = (index, field, value) => {
    const newHistory = [...history];
    newHistory[index] = {
      ...newHistory[index],
      [field]: field === 'date' ? value : parseFloat(value) || 0
    };
    setHistory(newHistory);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    
    const validHistory = history.filter(row => 
      row.date && row.open > 0 && row.high > 0 && row.low > 0 && row.close > 0
    );
    
    if (validHistory.length === 0) {
      alert('Please provide at least one valid historical data row');
      return;
    }
    
    onSubmit({
      periods: parseInt(periods),
      history: validHistory
    });
  };

  return (
    <div className="card">
      <h3>Price Forecasting</h3>
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="periods" className="form-label">Number of Days to Forecast</label>
          <input
            type="number"
            id="periods"
            min="1"
            max="30"
            value={periods}
            onChange={(e) => setPeriods(e.target.value)}
            placeholder="Enter the number of future days to predict (1-30)"
            className="form-input"
          />
        </div>

        <div className="historical-data-section">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
            <h4 style={{ margin: 0 }}>Historical OHLCV Data (oldest to newest)</h4>
            <div>
              <button 
                type="button" 
                onClick={() => updateHistoryCount(historyCount - 1)}
                disabled={historyCount <= 1}
                className="btn"
                style={{ marginRight: '10px', padding: '5px 10px', fontSize: '12px', background: '#6b7280', color: 'white' }}
              >
                Remove Row
              </button>
              <button 
                type="button" 
                onClick={() => updateHistoryCount(historyCount + 1)}
                disabled={historyCount >= 20}
                className="btn"
                style={{ padding: '5px 10px', fontSize: '12px', background: '#6b7280', color: 'white' }}
              >
                Add Row
              </button>
              <span style={{ marginLeft: '10px', fontSize: '14px', color: '#666' }}>
                {historyCount} rows
              </span>
            </div>
          </div>
          <div className="table-container" style={{ overflowX: 'auto', border: '1px solid #e2e8f0', borderRadius: '6px' }}>
            <table className="table" style={{ minWidth: '700px' }}>
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Open</th>
                  <th>High</th>
                  <th>Low</th>
                  <th>Close</th>
                  <th>Volume</th>
                </tr>
              </thead>
              <tbody>
                {history.map((row, index) => (
                  <tr key={index}>
                    <td>
                      <input
                        type="date"
                        value={row.date}
                        onChange={(e) => updateHistoryRow(index, 'date', e.target.value)}
                        required
                      />
                    </td>
                    <td>
                      <input
                        type="number"
                        step="0.01"
                        min="0"
                        value={row.open}
                        onChange={(e) => updateHistoryRow(index, 'open', e.target.value)}
                        required
                      />
                    </td>
                    <td>
                      <input
                        type="number"
                        step="0.01"
                        min="0"
                        value={row.high}
                        onChange={(e) => updateHistoryRow(index, 'high', e.target.value)}
                        required
                      />
                    </td>
                    <td>
                      <input
                        type="number"
                        step="0.01"
                        min="0"
                        value={row.low}
                        onChange={(e) => updateHistoryRow(index, 'low', e.target.value)}
                        required
                      />
                    </td>
                    <td>
                      <input
                        type="number"
                        step="0.01"
                        min="0"
                        value={row.close}
                        onChange={(e) => updateHistoryRow(index, 'close', e.target.value)}
                        required
                      />
                    </td>
                    <td>
                      <input
                        type="number"
                        min="0"
                        value={row.volume}
                        onChange={(e) => updateHistoryRow(index, 'volume', e.target.value)}
                        required
                      />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <button type="submit" disabled={loading} className="btn btn-primary" style={{ padding: '12px 24px', fontSize: '16px' }}>
          {loading ? 'Generating Forecast...' : 'Generate Forecast'}
        </button>
      </form>
    </div>
  );
};

const ForecastResults = ({ forecast, loading, error }) => {
  if (loading) {
    return (
      <div className="card">
        <h3>Forecast Results</h3>
        <div style={{ textAlign: 'center', padding: '40px' }}>
          <span className="loading"></span>
          <div>Loading forecast...</div>
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

const ForecastTable = ({ forecast, loading, error, onForecastSubmit }) => {
  return (
    <div className="container">
      <HistoricalDataForm onSubmit={onForecastSubmit} loading={loading} />
      <ForecastResults forecast={forecast} loading={loading} error={error} />
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
  error: PropTypes.string,
  onForecastSubmit: PropTypes.func.isRequired
};export default ForecastTable;