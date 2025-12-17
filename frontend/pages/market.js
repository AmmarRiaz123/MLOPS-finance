import { useState } from 'react';
import Head from 'next/head';
import { analyzeMarket } from '../lib/market';

export default function MarketAnalyzePage() {
  const [analysisWindow, setAnalysisWindow] = useState(30);
  const [forecastPeriod, setForecastPeriod] = useState(7);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const onSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setResult(null);

    try {
      const res = await analyzeMarket({
        analysis_window: parseInt(analysisWindow, 10),
        forecast_period: parseInt(forecastPeriod, 10),
      });
      setResult(res);
    } catch (err) {
      setError(err.message || 'Market analysis failed');
    } finally {
      setLoading(false);
    }
  };

  const Img = ({ title, b64 }) => {
    if (!b64) return null;
    return (
      <div className="card">
        <h3>{title}</h3>
        {/* base64 png from backend */}
        <img
          src={`data:image/png;base64,${b64}`}
          alt={title}
          style={{ width: '100%', marginTop: '1rem', borderRadius: '0.5rem' }}
        />
      </div>
    );
  };

  return (
    <>
      <Head>
        <title>Market Analysis - MLOps Finance</title>
      </Head>

      <div className="container">
        <div className="card">
          <h1 style={{ marginBottom: '0.75rem', fontSize: '1.5rem', fontWeight: 700 }}>
            Market Analysis
          </h1>
          <p style={{ color: '#6b7280' }}>
            One-call orchestration: loads data from backend CSVs, runs models, and returns plots.
          </p>

          <form onSubmit={onSubmit} style={{ marginTop: '1rem' }}>
            <div className="grid grid-cols-2">
              <div className="form-group">
                <label className="form-label" htmlFor="analysis_window">Analysis Window (rows)</label>
                <input
                  id="analysis_window"
                  className="form-input"
                  type="number"
                  min="5"
                  max="365"
                  value={analysisWindow}
                  onChange={(e) => setAnalysisWindow(e.target.value)}
                  required
                />
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="forecast_period">Forecast Period (days)</label>
                <input
                  id="forecast_period"
                  className="form-input"
                  type="number"
                  min="1"
                  max="30"
                  value={forecastPeriod}
                  onChange={(e) => setForecastPeriod(e.target.value)}
                  required
                />
              </div>
            </div>

            <button className="btn btn-primary" type="submit" disabled={loading}>
              {loading ? <span className="loading"></span> : 'Run Analysis'}
            </button>
          </form>

          {error && <div className="error">{error}</div>}
        </div>

        {result?.metrics && (
          <div className="card">
            <h3>Metrics</h3>
            <pre style={{ marginTop: '1rem', whiteSpace: 'pre-wrap' }}>
              {JSON.stringify(result.metrics, null, 2)}
            </pre>
          </div>
        )}

        {result?.plots && (
          <>
            <Img title="Price Trend" b64={result.plots.price_trend} />
            <Img title="Returns" b64={result.plots.returns} />
            <Img title="Volatility" b64={result.plots.volatility} />
            <Img title="Market Regime" b64={result.plots.regime} />
            <Img title="Forecast" b64={result.plots.forecast} />
          </>
        )}
      </div>
    </>
  );
}
