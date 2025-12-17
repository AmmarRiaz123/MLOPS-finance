import apiClient from './apiClient';

export const analyzeMarket = async ({ analysis_window, forecast_period }) => {
  return apiClient.post('/market/analyze', { analysis_window, forecast_period });
};
