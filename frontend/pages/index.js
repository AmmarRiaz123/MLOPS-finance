import { useState } from 'react';
import Head from 'next/head';
import OHLCVForm from '../components/OHLCVForm';
import PredictionCard from '../components/PredictionCard';
import { predictReturnLightGBM, predictReturnRandomForest } from '../lib/return';
import { predictDirection } from '../lib/direction';
import { predictVolatility } from '../lib/volatility';

export default function Dashboard() {
  const [loading, setLoading] = useState(false);
  const [predictions, setPredictions] = useState({});
  const [errors, setErrors] = useState({});

  const handleOHLCVSubmit = async (ohlcvData) => {
    setLoading(true);
    setErrors({});
    
    const newPredictions = {};
    const newErrors = {};

    // Run all OHLCV-based predictions
    const predictors = [
      { key: 'lightgbm_return', fn: () => predictReturnLightGBM(ohlcvData) },
      { key: 'rf_return', fn: () => predictReturnRandomForest(ohlcvData) },
      { key: 'direction', fn: () => predictDirection(ohlcvData) },
      { key: 'volatility', fn: () => predictVolatility(ohlcvData) }
    ];

    await Promise.allSettled(
      predictors.map(async ({ key, fn }) => {
        try {
          const result = await fn();
          newPredictions[key] = result;
        } catch (error) {
          newErrors[key] = error.message;
        }
      })
    );

    setPredictions(newPredictions);
    setErrors(newErrors);
    setLoading(false);
  };

  return (
    <>
      <Head>
        <title>MLOps Finance Dashboard</title>
      </Head>

      <div>
        <h1 style={{ marginBottom: '2rem', fontSize: '2rem', fontWeight: 'bold' }}>
          Market Analytics Dashboard
        </h1>

        <div className="grid grid-cols-1">
          <OHLCVForm onSubmit={handleOHLCVSubmit} loading={loading} />
          
          {(Object.keys(predictions).length > 0 || Object.keys(errors).length > 0) && (
            <div>
              <h2 style={{ marginBottom: '1rem', fontSize: '1.25rem', fontWeight: '600' }}>
                Predictions
              </h2>
              
              <div className="grid grid-cols-2">
                <PredictionCard
                  model="LightGBM Return"
                  prediction={predictions.lightgbm_return?.predicted_return?.toFixed(6)}
                  error={errors.lightgbm_return}
                />
                
                <PredictionCard
                  model="Random Forest Return"
                  prediction={predictions.rf_return?.predicted_return?.toFixed(6)}
                  error={errors.rf_return}
                />
                
                <PredictionCard
                  model="Direction"
                  prediction={predictions.direction?.direction}
                  probability={predictions.direction?.probability}
                  error={errors.direction}
                />
                
                <PredictionCard
                  model="Volatility"
                  prediction={predictions.volatility?.volatility?.toFixed(6)}
                  error={errors.volatility}
                />
              </div>
            </div>
          )}

          <div className="card" style={{ marginTop: '2rem' }}>
            <h3>Additional Analysis</h3>
            <p style={{ color: '#6b7280', marginTop: '0.5rem' }}>
              For forecasting and regime analysis, visit the dedicated pages using the navigation above.
              These require different input formats (time periods for forecasting, returns/volatility windows for regime detection).
            </p>
          </div>
        </div>
      </div>
    </>
  );
}
