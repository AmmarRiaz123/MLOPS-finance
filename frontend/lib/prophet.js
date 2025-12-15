import apiClient from './apiClient';

export const forecastPrice = async (forecastData) => {
  return apiClient.post('/forecast/price', forecastData);
};
