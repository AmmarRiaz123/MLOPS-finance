import apiClient from './apiClient';

export const predictReturnLightGBM = async (ohlcvData) => {
  return apiClient.post('/predict/return/lightgbm', ohlcvData);
};

export const predictReturnRandomForest = async (ohlcvData) => {
  return apiClient.post('/predict/return/random-forest', ohlcvData);
};
