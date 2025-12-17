# Models — Input requirements & provenance

This document lists each model in training/, the exact input features used during training (when available), and where the feature list was obtained.

Important: inference must reproduce these feature vectors exactly (names + order). Preferred canonical sources:
- models/latest/feature_names.json (saved next to deployed model)
- training/<model>/metrics/latest/training_features.json (saved by training scripts)
- training/<model>/features.py (candidate_features / prepare_* outputs)

---

## 1) LightGBM return model
- Path: training/lightgbm_return/
- Canonical feature list (saved to training metrics and model dir):
  - return_lag1
  - return_lag2
  - ret_3
  - ret_5
  - ret_10
  - ret_mean_5
  - ret_std_5
  - momentum_8
  - rsi_14
  - vol_ma5
  - high_low_spread
  - vol_x_std5
- Files (authoritative):
  - training/lightgbm_return/metrics/latest/training_features.json
  - models/latest/feature_names.json

## 2) Random Forest return model
- Path: training/random_forest_return/
- Canonical feature list:
  - return_lag1
  - return_lag2
  - return_lag3
  - return_lag5
  - return_lag10
  - ma5
  - ma10
  - ma20
  - std5
  - std10
  - std20
  - momentum_8
  - vol_ma5
  - vol_ma10
  - high_low_spread
  - open_close_spread
  - vol_x_std5
  - rsi_14
  - macd
  - macd_signal
  - stoch_k
  - stoch_d
- Files (authoritative):
  - training/random_forest_return/metrics/latest/training_features.json
  - models/latest/feature_names.json

## 3) LightGBM up/down (direction) model
- Path: training/lightgbm_up_down/
- Canonical feature list (saved by training):
  - See training/lightgbm_up_down/metrics/latest/training_features.json or models/latest/feature_names.json for the full ordered list (training reports `features_count`: 39).
- Top features (from training metrics):
  - return_skew_20
  - daily_return
  - return_kurt_5
  - close_to_ma10
  - return_mean_20
- Files (authoritative):
  - training/lightgbm_up_down/metrics/latest/training_features.json
  - models/latest/feature_names.json

## 4) LightGBM volatility model
- Path: training/lightgbm_volatility/
- Canonical feature list (saved by training):
  - ret_lag1
  - ret_lag2
  - ret_lag3
  - ret_mean_3
  - ret_std_3
  - ma5
  - ma10
  - std5
  - std10
  - high_low_spread
  - open_close_spread
  - vol_ma5
  - vol_ma10
  - ewma_8
  - ewma_16
  - momentum_8
  - rsi_14
  - macd
  - macd_signal
  - stoch_k
  - stoch_d
  - vol_x_std5
  - vol_x_mom8
- Files (authoritative):
  - training/lightgbm_volatility/metrics/latest/training_features.json
  - models/latest/feature_names.json

## 5) Prophet forecast model
- Path: training/prophet_forecast/
- Required columns in training DataFrame:
  - ds (datetime)
  - y (target; training uses log(Close) as y)
  - regressors added during training (canonical regressors saved):
    - volume
    - high_low_spread
    - open_close_spread
- Files (authoritative):
  - training/prophet_forecast/metrics/latest/training_metadata.json  (contains regressors)
  - training/prophet_forecast/metrics/latest/training_features.json
  - models/latest/feature_names.json (if present)
- Note:
  - For forecasting, future DataFrame must include `ds` plus all regressors above. The service auto-fills regressors from history or uses defaults.

## 6) Market regime HMM model
- Path: training/market_regime_hmm/
- Canonical feature list (saved by training):
  - See training/market_regime_hmm/metrics/latest/training_features.json or models/latest/feature_names.json for the full ordered list used to fit the HMM/GMM.
- Inference API expects:
  - returns_window: List[float]
  - volatility_window: List[float]
  - The regime service maps these windows into the model input vector using the same engineered pipeline as training.
- Files (authoritative):
  - training/market_regime_hmm/metrics/latest/training_features.json
  - models/latest/feature_names.json

---

## Persisted canonical feature lists

Each training script persists the canonical feature-name list in two places after a successful run:

- models/latest/feature_names.json — canonical ordered list saved next to the deployed model
- training/<model>/metrics/latest/training_features.json — training metrics directory copy for discoverability

If you do not see a feature list for a model, re-run the corresponding training script from the project root to generate these files:
- python training/lightgbm_up_down/train.py
- python training/market_regime_hmm/train.py
- (or re-run any model's train.py after updating data)

After training completes, verify the saved lists in:
- models/latest/feature_names.json
- training/<model>/metrics/latest/training_features.json

Use these canonical files as the authoritative source for inference feature ordering.

---

## How to use these lists in inference

- Implement app.services.feature_mapper to:
  - load models/latest/feature_names.json and any saved scaler,
  - accept raw OHLCV and optional short history,
  - compute and order engineered features exactly as training,
  - validate the produced vector matches the saved ordering before calling model.predict.

--- 

## Action items (recommended)
- Ensure all training scripts write models/latest/feature_names.json and training/<model>/metrics/latest/training_features.json (already added to trains).
- Implement app.services.feature_mapper and update model services to require canonical feature vectors (or accept raw OHLCV + history and call the mapper).
- Add CI check to validate that for each deployed model the pair models/latest/feature_names.json and training/<model>/metrics/latest/training_features.json agree.
