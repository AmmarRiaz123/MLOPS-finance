# Model Metrics (Latest)

This page summarizes the **latest** model artifacts and their most recent evaluation metrics.
Source-of-truth is always the corresponding file under `training/<model>/metrics/latest/`.

> Note: Some metrics files were not provided in the current repo snapshot, so those entries are left as **TODO** with the exact file path you should copy from.

---

## 1) Return — LightGBM (`lightgbm_return_model.pkl`)
{
  "trained_at": "2025-12-17T18:15:41.227477",
  "n_train": 12876,
  "n_val": 3219,
  "features": [
    "ret_3",
    "ret_mean_5",
    "rsi_14",
    "vol_x_std5"
  ],
  "params": {
    "num_leaves": 12,
    "max_depth": 3,
    "learning_rate": 0.01,
    "n_estimators": 100,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "min_data_in_leaf": 20,
    "min_gain_to_split": 0.01,
    "boosting_type": "gbdt"
  },
  "val_rmse": 0.02551312315095833,
  "val_mae": 0.01781476408814016,
  "feature_importance": {
    "ret_3": 18.83707,
    "ret_mean_5": 40.102965,
    "rsi_14": 41.059965,
    "vol_x_std5": 0.0
  }
}
---

## 2) Return — Random Forest (`random_forest_return_model.pkl`)
{
  "trained_at": "2025-12-17T18:15:49.410286",
  "horizon": 1,
  "n_train": 12877,
  "n_val": 3220,
  "params": {
    "n_estimators": 200,
    "max_depth": null,
    "min_samples_split": 2,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1
  },
  "val_rmse": 0.017995339362747245,
  "val_mae": 0.012827548324758561,
  "feature_importance": {
    "return_lag1": 4.820145,
    "return_lag2": 4.706751,
    "return_lag3": 5.367656,
    "return_lag5": 4.867276,
    "return_lag10": 4.870363,
    "ma5": 3.716282,
    "ma10": 3.769757,
    "ma20": 3.729204,
    "std5": 4.537371,
    "std10": 4.684196,
    "std20": 4.702284,
    "momentum_8": 5.479853,
    "vol_ma5": 4.100154,
    "vol_ma10": 3.980057,
    "high_low_spread": 5.703469,
    "open_close_spread": 5.167313,
    "vol_x_std5": 4.190231,
    "rsi_14": 4.259634,
    "macd": 4.518582,
    "macd_signal": 4.60531,
    "stoch_k": 4.06289,
    "stoch_d": 4.161221
  }
}
---

## 3) Direction — LightGBM Up/Down (`lightgbm_up_down_model.pkl`)
{
  "data_file": "C:\\Users\\Dell\\Desktop\\MLOPS-finance\\mlops-project\\data\\IBM_yfinance_1d.csv",
  "train_samples": 13399,
  "test_samples": 2679,
  "features_count": 39,
  "test_accuracy": 0.48040313549832026,
  "feature_importance_top5": {
    "return_mean_20": 19,
    "return_lag3": 19,
    "return_lag1": 19,
    "return_mean_5": 18,
    "return_mean_10": 15
  }
}
---

## 4) Volatility — LightGBM (`lightgbm_volatility_model.pkl`)
{
  "trained_at": "2025-12-17T18:15:53.501300",
  "n_train": 12876,
  "n_val": 3220,
  "params": {
    "num_leaves": 16,
    "max_depth": 6,
    "learning_rate": 0.03,
    "n_estimators": 120,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "min_data_in_leaf": 20,
    "boosting_type": "gbdt",
    "random_state": 42
  },
  "val_rmse": 0.00915894127168337,
  "val_mae": 0.006039734547544513,
  "feature_importance": {
    "ret_lag1": 0.354393,
    "ret_lag2": 0.4978,
    "ret_lag3": 0.901617,
    "ret_mean_3": 0.876144,
    "ret_std_3": 0.812156,
    "ma5": 2.549479,
    "ma10": 1.713934,
    "std5": 3.047539,
    "std10": 37.002362,
    "high_low_spread": 19.216426,
    "open_close_spread": 1.11922,
    "vol_ma5": 1.360385,
    "vol_ma10": 1.685799,
    "ewma_8": 1.384594,
    "ewma_16": 3.377403,
    "momentum_8": 4.424536,
    "rsi_14": 1.248553,
    "macd": 6.792811,
    "macd_signal": 2.584877,
    "stoch_k": 0.684775,
    "stoch_d": 1.678322,
    "vol_x_std5": 2.685565,
    "vol_x_mom8": 4.001311
  }
}
---

## 5) Forecast — Prophet (`prophet_forecast.pkl`)
{
  "trained_at": "2025-12-17T18:16:38.212778",
  "data_start": "1962-01-02T00:00:00",
  "data_end": "2025-12-16T00:00:00",
  "train_end": "2013-02-28T00:00:00",
  "n_train": 12878,
  "n_test": 3220,
  "regressors": [
    "volume",
    "high_low_spread",
    "open_close_spread"
  ],
  "target": "Close",
  "target_transform": "log",
  "params": {
    "selected_changepoint_prior_scale": 0.2,
    "candidate_changepoint_prior_scales": [
      0.01,
      0.05,
      0.1,
      0.2
    ],
    "interval_width": 0.8,
    "monthly_seasonality": {
      "period": 30.5,
      "fourier_order": 5
    }
  },
  "validation_rmse": 392.19133638022237,
  "regressor_coefs": {
    "intercept": 2.303652194552101,
    "volume": -0.0050210732045783265,
    "high_low_spread": -0.00041429917313463935,
    "open_close_spread": -0.002623155960622273
  }
}

---

## 6) Regime — HMM / GMM (`market_regime_hmm.pkl`)
{
  "trained_at": "2025-12-17T18:16:43.221936",
  "n_components": 3,
  "regime_counts": {
    "0": 7765,
    "1": 3347,
    "2": 4986
  },
  "avg_return_per_regime": {
    "0": 0.0005074912824700867,
    "1": 0.00040437906631350884,
    "2": 7.872702198498192e-06
  },
  "avg_vol_per_regime": {
    "0": 0.010287009813843322,
    "1": 0.023310020335371585,
    "2": 0.011289004970760587
  },
  "transition_matrix": [
    [
      0.933419188667096,
      0.06130070830650354,
      0.005280103026400515
    ],
    [
      0.14375373580394502,
      0.8368200836820083,
      0.019426180514046622
    ],
    [
      0.007220216606498195,
      0.014239871640593663,
      0.9785399117529081
    ]
  ],
  "hmm_backend": "gmm_fallback"
}
---
