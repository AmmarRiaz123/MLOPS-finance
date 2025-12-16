import { useState } from 'react';
import Head from 'next/head';
import ForecastTable from '../components/ForecastTable';
import { forecastPrice } from '../lib/prophet';

export default function Forecast() {
  const [loading, setLoading] = useState(false);
  const [forecast, setForecast] = useState(null);
  const [error, setError] = useState('');

  const handleForecastSubmit = async (data) => {
    setLoading(true);
    setError('');
    setForecast(null);
    
    try {
      const result = await forecastPrice(data);
      setForecast(result.forecast);
    } catch (err) {
      setError(err.message || 'Forecast failed');
      setForecast(null);
    }
    
    setLoading(false);
  };

  return (
    <>
      <Head>
        <title>Price Forecast - MLOps Finance</title>
      </Head>

      <ForecastTable 
        forecast={forecast} 
        loading={loading} 
        error={error}
        onForecastSubmit={handleForecastSubmit}
      />
    </>
  );
}
