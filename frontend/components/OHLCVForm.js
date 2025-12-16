import { useState } from 'react';
import PropTypes from 'prop-types';

const OHLCVForm = ({ onSubmit, loading = false }) => {
  const [formData, setFormData] = useState({
    open: '',
    high: '',
    low: '',
    close: '',
    volume: ''
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    
    // Convert to numbers
    const numericData = {
      open: parseFloat(formData.open),
      high: parseFloat(formData.high),
      low: parseFloat(formData.low),
      close: parseFloat(formData.close),
      volume: parseFloat(formData.volume)
    };

    // Basic validation
    const values = Object.values(numericData);
    if (values.some(val => isNaN(val) || val <= 0)) {
      alert('Please enter valid positive numbers for all fields');
      return;
    }

    if (numericData.high < numericData.low) {
      alert('High price must be greater than or equal to low price');
      return;
    }

    onSubmit(numericData);
  };

  return (
    <form onSubmit={handleSubmit} className="card">
      <h3>Market Data Input</h3>
      
      <div className="grid grid-cols-2">
        <div className="form-group">
          <label htmlFor="open" className="form-label">Open Price</label>
          <input
            id="open"
            name="open"
            type="number"
            step="0.01"
            value={formData.open}
            onChange={handleChange}
            className="form-input"
            placeholder="100.00"
            required
          />
        </div>

        <div className="form-group">
          <label htmlFor="high" className="form-label">High Price</label>
          <input
            id="high"
            name="high"
            type="number"
            step="0.01"
            value={formData.high}
            onChange={handleChange}
            className="form-input"
            placeholder="101.50"
            required
          />
        </div>

        <div className="form-group">
          <label htmlFor="low" className="form-label">Low Price</label>
          <input
            id="low"
            name="low"
            type="number"
            step="0.01"
            value={formData.low}
            onChange={handleChange}
            className="form-input"
            placeholder="99.50"
            required
          />
        </div>

        <div className="form-group">
          <label htmlFor="close" className="form-label">Close Price</label>
          <input
            id="close"
            name="close"
            type="number"
            step="0.01"
            value={formData.close}
            onChange={handleChange}
            className="form-input"
            placeholder="100.75"
            required
          />
        </div>
      </div>

      <div className="form-group">
        <label htmlFor="volume" className="form-label">Volume</label>
        <input
          id="volume"
          name="volume"
          type="number"
          step="1"
          value={formData.volume}
          onChange={handleChange}
          className="form-input"
          placeholder="1000000"
          required
        />
      </div>

      <button
        type="submit"
        disabled={loading}
        className="btn btn-primary"
      >
        {loading ? <span className="loading"></span> : 'Predict'}
      </button>
    </form>
  );
};

OHLCVForm.propTypes = {
  onSubmit: PropTypes.func.isRequired,
  loading: PropTypes.bool
};

export default OHLCVForm;
