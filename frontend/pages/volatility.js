import { useState } from 'react';
import Head from 'next/head';
import OHLCVForm from '../components/OHLCVForm';
import PredictionCard from '../components/PredictionCard';
import { predictVolatility } from '../lib/volatility';

export default function Volatility() {
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState('');

  const handleSubmit = async (ohlcvData) => {
    setLoading(true);
    setError('');
    
    try {
      const result = await predictVolatility(ohlcvData);
      setPrediction(result);
    } catch (err) {
      setError(err.message);
      setPrediction(null);
    }
    
    setLoading(false);
  };

  return (
    <>
      <Head>
        <title>Volatility Prediction - MLOps Finance</title>
      </Head>

      <div>
        <h1 style={{ marginBottom: '2rem', fontSize: '2rem', fontWeight: 'bold' }}>
          Volatility Prediction
        </h1>

        <div className="grid grid-cols-1">
          <OHLCVForm onSubmit={handleSubmit} loading={loading} />
          
          {(prediction || error) && (
            <div>
              <h2 style={{ marginBottom: '1rem', fontSize: '1.25rem', fontWeight: '600' }}>
                Volatility Prediction
              </h2>
              
              <PredictionCard
                model={prediction?.model || 'Volatility Model'}
                prediction={prediction?.volatility?.toFixed(6)}
                error={error}
              />
            </div>
          )}
        </div>
      </div>
    </>
  );
}
