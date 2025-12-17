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

  const formatPct = (x, digits = 4) => {
    const n = Number(x);
    if (!Number.isFinite(n)) return '-';
    return `${(n * 100).toFixed(digits)}%`;
  };

  const formatNum = (x, digits = 6) => {
    const n = Number(x);
    if (!Number.isFinite(n)) return '-';
    return n.toFixed(digits);
  };

  const directionLabel = (d) => {
    // your API currently returns "0"/"1" sometimes; make it user-friendly
    const s = String(d ?? '').toLowerCase();
    if (s === '1' || s === 'up' || s === 'bull') return 'Up';
    if (s === '0' || s === 'down' || s === 'bear') return 'Down';
    return String(d ?? '-');
  };

  const Badge = ({ text, tone = 'neutral' }) => (
    <span className={`badge badge-${tone}`}>{text}</span>
  );

  const MetricCard = ({ title, value, subtitle, tone = 'neutral' }) => (
    <div className="card metric-card">
      <div className="metric-title">{title}</div>
      <div className="metric-value">
        <span className={`metric-value-text metric-${tone}`}>{value}</span>
      </div>
      {subtitle ? <div className="metric-subtitle">{subtitle}</div> : null}
    </div>
  );

  const PlotCard = ({ title, b64 }) => {
    if (!b64) return null;
    return (
      <div className="card plot-card">
        <div className="plot-title">{title}</div>
        <img
          src={`data:image/png;base64,${b64}`}
          alt={title}
          className="plot-img"
        />
      </div>
    );
  };

  const metrics = result?.metrics;
  const plots = result?.plots;

  const retLgb = metrics?.return?.lightgbm_return_model;
  const retRf = metrics?.return?.random_forest_return_model;
  const dir = metrics?.direction;
  const vol = metrics?.volatility?.value;
  const reg = metrics?.regime;
  const fc = metrics?.forecast;

  // simple forecast summary
  const forecastPoints = fc?.points || [];
  const nextPoint = forecastPoints[0];
  const minY = forecastPoints.length ? Math.min(...forecastPoints.map(p => Number(p.yhat)).filter(Number.isFinite)) : null;
  const maxY = forecastPoints.length ? Math.max(...forecastPoints.map(p => Number(p.yhat)).filter(Number.isFinite)) : null;

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

        {metrics && (
          <>
            <div className="section-header">
              <h2 className="section-title">Key Metrics</h2>
              <div className="section-subtitle">
                Window: <b>{result.analysis_window}</b> rows, Forecast: <b>{result.forecast_period}</b> days
              </div>
            </div>

            <div className="grid grid-cols-3 metrics-grid">
              <MetricCard
                title="Return (LightGBM)"
                value={formatPct(retLgb, 4)}
                subtitle={`raw: ${formatNum(retLgb, 6)}`}
                tone={Number(retLgb) >= 0 ? 'bull' : 'bear'}
              />
              <MetricCard
                title="Return (Random Forest)"
                value={formatPct(retRf, 4)}
                subtitle={`raw: ${formatNum(retRf, 6)}`}
                tone={Number(retRf) >= 0 ? 'bull' : 'bear'}
              />
              <MetricCard
                title="Volatility"
                value={formatNum(vol, 6)}
                subtitle="model output (unit depends on training target)"
                tone="neutral"
              />

              <div className="card metric-card">
                <div className="metric-title">Direction</div>
                <div className="metric-value" style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                  <span className="metric-value-text">{directionLabel(dir?.direction)}</span>
                  <Badge
                    text={`p=${Number.isFinite(Number(dir?.probability)) ? Number(dir.probability).toFixed(3) : '-'}`}
                    tone={directionLabel(dir?.direction) === 'Up' ? 'bull' : 'bear'}
                  />
                </div>
                <div className="metric-subtitle">classification result</div>
              </div>

              <div className="card metric-card">
                <div className="metric-title">Regime</div>
                <div className="metric-value" style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                  <span className="metric-value-text">{String(reg?.regime_label ?? '-')}</span>
                  <Badge
                    text={`score=${Number.isFinite(Number(reg?.score)) ? Number(reg.score).toFixed(3) : '-'}`}
                    tone={String(reg?.regime_label ?? '').toLowerCase().includes('bull') ? 'bull'
                      : String(reg?.regime_label ?? '').toLowerCase().includes('bear') ? 'bear'
                      : 'neutral'}
                  />
                </div>
                <div className="metric-subtitle">{`id=${reg?.regime_id ?? '-'}`}</div>
              </div>

              <div className="card metric-card">
                <div className="metric-title">Forecast (summary)</div>
                <div className="metric-subtitle">
                  {nextPoint
                    ? `next: ${new Date(nextPoint.ds).toLocaleDateString()}  yhat=${formatNum(nextPoint.yhat, 4)}`
                    : 'no forecast points'}
                </div>
                <div className="metric-subtitle">
                  {(minY != null && maxY != null)
                    ? `range: ${formatNum(minY, 4)} → ${formatNum(maxY, 4)}`
                    : ''}
                </div>
              </div>
            </div>

            <div className="card">
              <details>
                <summary style={{ cursor: 'pointer', fontWeight: 600 }}>Raw metrics JSON</summary>
                <pre className="code-block">{JSON.stringify(metrics, null, 2)}</pre>
              </details>
            </div>
          </>
        )}

        {plots && (
          <>
            <div className="section-header">
              <h2 className="section-title">Plots</h2>
              <div className="section-subtitle">Rendered server-side (PNG base64)</div>
            </div>

            <div className="grid grid-cols-2">
              <PlotCard title="Price Trend" b64={plots.price_trend} />
              <PlotCard title="Returns" b64={plots.returns} />
              <PlotCard title="Volatility" b64={plots.volatility} />
              <PlotCard title="Market Regime" b64={plots.regime} />
              <div className="grid-col-span-2">
                <PlotCard title="Forecast" b64={plots.forecast} />
              </div>
            </div>
          </>
        )}
      </div>
    </>
  );
}
