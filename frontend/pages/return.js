import { useState } from 'react';
import Head from 'next/head';
import OHLCVForm from '../components/OHLCVForm';
import PredictionCard from '../components/PredictionCard';
import { predictReturnLightGBM, predictReturnRandomForest } from '../lib/return';

export default function Return() {
  const [loading, setLoading] = useState(false);
  const [predictions, setPredictions] = useState({});
  const [errors, setErrors] = useState({});

  const handleSubmit = async (ohlcvData) => {
    setLoading(true);
    setErrors({});
    
    const newPredictions = {};
    const newErrors = {};

    try {
      const [lgbmResult, rfResult] = await Promise.allSettled([
        predictReturnLightGBM(ohlcvData),
        predictReturnRandomForest(ohlcvData)
      ]);

      if (lgbmResult.status === 'fulfilled') {
        newPredictions.lightgbm = lgbmResult.value;
      } else {
        newErrors.lightgbm = lgbmResult.reason.message;
      }

      if (rfResult.status === 'fulfilled') {
        newPredictions.randomForest = rfResult.value;
      } else {
        newErrors.randomForest = rfResult.reason.message;
      }
    } catch (error) {
      newErrors.general = error.message;
    }

    setPredictions(newPredictions);
    setErrors(newErrors);
    setLoading(false);
  };

  return (
    <>
      <Head>
        <title>Return Prediction - MLOps Finance</title>
      </Head>

      <div>
        <h1 style={{ marginBottom: '2rem', fontSize: '2rem', fontWeight: 'bold' }}>
          Return Prediction
        </h1>

        <div className="grid grid-cols-1">
          <OHLCVForm onSubmit={handleSubmit} loading={loading} />
          
          {(Object.keys(predictions).length > 0 || Object.keys(errors).length > 0) && (
            <div>
              <h2 style={{ marginBottom: '1rem', fontSize: '1.25rem', fontWeight: '600' }}>
                Return Predictions
              </h2>
              
              <div className="grid grid-cols-2">
                <PredictionCard
                  model="LightGBM"
                  prediction={predictions.lightgbm?.predicted_return?.toFixed(6)}
                  error={errors.lightgbm}
                />
                
                <PredictionCard
                  model="Random Forest"
                  prediction={predictions.randomForest?.predicted_return?.toFixed(6)}
                  error={errors.randomForest}
                />
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
