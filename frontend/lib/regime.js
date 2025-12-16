import apiClient from './apiClient';

export const predictRegime = async (regimeData) => {
  return apiClient.post('/predict/regime', regimeData);
};
