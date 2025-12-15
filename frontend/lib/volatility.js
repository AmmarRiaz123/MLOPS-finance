import apiClient from './apiClient';

export const predictVolatility = async (ohlcvData) => {
  return apiClient.post('/predict/volatility', ohlcvData);
};
