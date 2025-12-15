import { useState } from 'react';
import Head from 'next/head';
import ForecastTable from '../components/ForecastTable';
import { forecastPrice } from '../lib/prophet';

export default function Forecast() {
  const [periods, setPeriods] = useState('5');
  const [loading, setLoading] = useState(false);
  const [forecast, setForecast] = useState(null);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    const numPeriods = parseInt(periods);
    if (isNaN(numPeriods) || numPeriods < 1 || numPeriods > 30) {
      setError('Please enter a valid number of periods (1-30)');
      return;
    }

    setLoading(true);
    setError('');
    
    try {
      const result = await forecastPrice({ periods: numPeriods });
      setForecast(result.forecast);
    } catch (err) {
      setError(err.message);
      setForecast(null);
    }
    
    setLoading(false);
  };

  return (
    <>
      <Head>
        <title>Price Forecast - MLOps Finance</title>
      </Head>

      <div>
        <h1 style={{ marginBottom: '2rem', fontSize: '2rem', fontWeight: 'bold' }}>
          Price Forecasting
        </h1>

        <form onSubmit={handleSubmit} className="card">
          <h3>Forecast Parameters</h3>
          
          <div className="form-group">
            <label htmlFor="periods" className="form-label">
              Number of Days to Forecast
            </label>
            <input
              id="periods"
              type="number"
              min="1"
              max="30"
              value={periods}
              onChange={(e) => setPeriods(e.target.value)}
              className="form-input"
              placeholder="5"
              required
            />
            <div style={{ fontSize: '0.875rem', color: '#6b7280', marginTop: '0.25rem' }}>
              Enter the number of future days to predict (1-30)
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="btn btn-primary"
          >
            {loading ? <span className="loading"></span> : 'Generate Forecast'}
          </button>
        </form>

        <ForecastTable 
          forecast={forecast} 
          loading={loading} 
          error={error} 
        />
      </div>
    </>
  );
}
