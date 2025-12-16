import apiClient from './apiClient';

export const predictDirection = async (ohlcvData) => {
  return apiClient.post('/predict/direction', ohlcvData);
};
