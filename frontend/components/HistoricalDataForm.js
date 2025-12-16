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
      // Add new empty rows
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
      // Remove excess rows
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
    
    // Validate data
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
          <label htmlFor="periods">Number of Days to Forecast</label>
          <input
            type="number"
            id="periods"
            min="1"
            max="30"
            value={periods}
            onChange={(e) => setPeriods(e.target.value)}
            placeholder="Enter the number of future days to predict (1-30)"
          />
        </div>

        <div className="form-group">
          <label htmlFor="historyCount">Number of Historical Days</label>
          <input
            type="number"
            id="historyCount"
            min="1"
            max="20"
            value={historyCount}
            onChange={(e) => updateHistoryCount(parseInt(e.target.value))}
            placeholder="Number of historical OHLCV rows to provide"
          />
        </div>

        <div className="historical-data-section">
          <h4>Historical OHLCV Data (oldest to newest)</h4>
          <div className="table-container">
            <table className="table historical-table">
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

        <button type="submit" disabled={loading} className="btn btn-primary">
          {loading ? 'Generating Forecast...' : 'Generate Forecast'}
        </button>
      </form>

      <style jsx>{`
        .historical-data-section {
          margin: 1rem 0;
        }
        
        .table-container {
          overflow-x: auto;
          margin: 1rem 0;
        }
        
        .historical-table {
          min-width: 700px;
        }
        
        .historical-table input {
          width: 100%;
          min-width: 80px;
          padding: 0.25rem;
          border: 1px solid #ddd;
          border-radius: 4px;
        }
        
        .historical-table input[type="date"] {
          min-width: 120px;
        }
        
        .historical-table input[type="number"] {
          text-align: right;
        }
        
        .form-group {
          margin-bottom: 1rem;
        }
        
        .form-group label {
          display: block;
          margin-bottom: 0.5rem;
          font-weight: bold;
        }
        
        .form-group input {
          width: 100%;
          padding: 0.5rem;
          border: 1px solid #ddd;
          border-radius: 4px;
        }
        
        .btn {
          padding: 0.75rem 1.5rem;
          border: none;
          border-radius: 4px;
          cursor: pointer;
          font-size: 1rem;
        }
        
        .btn-primary {
          background-color: #007bff;
          color: white;
        }
        
        .btn:disabled {
          background-color: #6c757d;
          cursor: not-allowed;
        }
        
        .card {
          background: white;
          border-radius: 8px;
          padding: 1.5rem;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
          margin-bottom: 1rem;
        }
        
        .table {
          width: 100%;
          border-collapse: collapse;
        }
        
        .table th,
        .table td {
          padding: 0.5rem;
          text-align: left;
          border-bottom: 1px solid #dee2e6;
        }
        
        .table th {
          background-color: #f8f9fa;
          font-weight: bold;
        }
      `}</style>
    </div>
  );
};

HistoricalDataForm.propTypes = {
  onSubmit: PropTypes.func.isRequired,
  loading: PropTypes.bool
};

export default HistoricalDataForm;
