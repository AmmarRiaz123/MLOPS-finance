import { useState } from 'react';
import Head from 'next/head';
import RegimeIndicator from '../components/RegimeIndicator';
import { predictRegime } from '../lib/regime';

export default function Regime() {
  const [returnsWindow, setReturnsWindow] = useState('0.001,-0.002,0.003,0.001,-0.001');
  const [volatilityWindow, setVolatilityWindow] = useState('0.01,0.012,0.011,0.013,0.009');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const parseArrayInput = (input) => {
    return input
      .split(',')
      .map(val => parseFloat(val.trim()))
      .filter(val => !isNaN(val));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    const returns = parseArrayInput(returnsWindow);
    const volatility = parseArrayInput(volatilityWindow);

    if (returns.length === 0 || volatility.length === 0) {
      setError('Please enter valid numeric values for both windows');
      return;
    }

    setLoading(true);
    setError('');
    
    try {
      const response = await predictRegime({
        returns_window: returns,
        volatility_window: volatility
      });
      setResult(response);
    } catch (err) {
      setError(err.message);
      setResult(null);
    }
    
    setLoading(false);
  };

  return (
    <>
      <Head>
        <title>Market Regime - MLOps Finance</title>
      </Head>

      <div>
        <h1 style={{ marginBottom: '2rem', fontSize: '2rem', fontWeight: 'bold' }}>
          Market Regime Analysis
        </h1>

        <form onSubmit={handleSubmit} className="card">
          <h3>Regime Analysis Input</h3>
          
          <div className="form-group">
            <label htmlFor="returns" className="form-label">
              Returns Window
            </label>
            <input
              id="returns"
              type="text"
              value={returnsWindow}
              onChange={(e) => setReturnsWindow(e.target.value)}
              className="form-input"
              placeholder="0.001,-0.002,0.003,0.001,-0.001"
              required
            />
            <div style={{ fontSize: '0.875rem', color: '#6b7280', marginTop: '0.25rem' }}>
              Comma-separated list of recent return values
            </div>
          </div>

          <div className="form-group">
            <label htmlFor="volatility" className="form-label">
              Volatility Window
            </label>
            <input
              id="volatility"
              type="text"
              value={volatilityWindow}
              onChange={(e) => setVolatilityWindow(e.target.value)}
              className="form-input"
              placeholder="0.01,0.012,0.011,0.013,0.009"
              required
            />
            <div style={{ fontSize: '0.875rem', color: '#6b7280', marginTop: '0.25rem' }}>
              Comma-separated list of recent volatility values
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="btn btn-primary"
          >
            {loading ? <span className="loading"></span> : 'Analyze Regime'}
          </button>
        </form>

        <RegimeIndicator
          regime={result?.regime}
          score={result?.score}
          model={result?.model}
          loading={loading}
          error={error}
        />
      </div>
    </>
  );
}
