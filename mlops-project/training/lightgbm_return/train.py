"""
Train LightGBM regressor for 1-day return prediction.
Saves model to mlops-project/models/latest/lightgbm_return.pkl
Saves metrics/metadata to training/lightgbm_return/metrics/latest/metrics.json
"""
from pathlib import Path
import json
import shutil
from datetime import datetime
import argparse
import joblib
import numpy as np
import os
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import pandas as pd

from features import prepare_features

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = REPO_ROOT / "models"
LATEST_MODEL_DIR = MODELS_ROOT / "latest"
LATEST_MODEL_PATH = LATEST_MODEL_DIR / "lightgbm_return_model.pkl"
ARCHIVED_MODELS = MODELS_ROOT / "archived"

METRICS_ROOT = Path(__file__).resolve().parent / "metrics"
METRICS_LATEST_DIR = METRICS_ROOT / "latest"
METRICS_ARCHIVED_DIR = METRICS_ROOT / "archived"

def _archive_if_exists(path: Path, archived_dir: Path, prefix: str):
    if path.exists():
        archived_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        dest = archived_dir / f"{ts}_{prefix}"
        shutil.move(str(path), str(dest))

def train(random_seed: int = 42,
          test_size_ratio: float = 0.2,
          params: dict = None,
          do_tune: bool = True,
          tune_budget: int = 10,
          target_horizon: int = 3):
    """
    Train with multi-day horizon support (target_horizon: 3 or 5).
    """
    # call prepare_features and enable scaling for numeric stability, request horizon
    res = prepare_features(scale=True, horizons=[target_horizon])  # returns (X,y,dates,scaler)
    # prepare_features may return (X,y,dates) or (X,y,dates,scaler)
    if len(res) == 4:
        X, y_all, dates, scaler = res
    else:
        X, y_all, dates = res
        scaler = None

    # pick target column (return_{target_horizon}d) — handle Series or DataFrame returns robustly
    target_col = f'return_{target_horizon}d'
    if isinstance(y_all, pd.Series):
        y = y_all
    elif isinstance(y_all, pd.DataFrame) and target_col in y_all.columns:
        y = y_all[target_col]
    else:
        # fallback: if y_all is a dict-like or single-column, try to coerce to Series
        try:
            y = pd.Series(y_all)
        except Exception:
            raise RuntimeError("Unable to resolve target Series from prepare_features() output")

    # time-based split (chronological)
    split_idx = int(len(X) * (1 - test_size_ratio))
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    dates_train, dates_val = dates.iloc[:split_idx], dates.iloc[split_idx:]

    # default params tuned for small data (prevent overfitting)
    default_params = {
        "num_leaves": 16,
        "max_depth": 6,
        "learning_rate": 0.03,
        "n_estimators": 120,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "random_state": random_seed,
        "n_jobs": -1,
        # regularization L1/L2
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        # stability defaults
        "min_data_in_leaf": 20,
        "min_gain_to_split": 0.01,
        "boosting_type": "gbdt"
    }
    model_params = default_params if params is None else {**default_params, **params}

    # Hyperparameter tuning (RandomizedSearchCV limited to small search)
    best_params = model_params.copy()
    if do_tune:
        from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
        from lightgbm import LGBMRegressor
        tss = TimeSeriesSplit(n_splits=3)
        # small targeted search to avoid overfitting / long runs
        param_dist = {
            "num_leaves": [12, 16, 20],
            "max_depth": [3,5,6],
            "learning_rate": [0.01,0.03],
            "n_estimators": [80,100,150],
            "feature_fraction": [0.7,0.8],
            "bagging_fraction": [0.7,0.8]
        }
        # try a small set of boosting types
        boosting_candidates = ['gbdt']
        best_score = None
        best_grid = None
        # use RandomizedSearchCV with limited iterations to speed up tuning
        n_iter_search = min(tune_budget, 8)
        for boost in boosting_candidates:
            estimator = LGBMRegressor(boosting_type=boost, random_state=random_seed, n_jobs=-1)
            rs = RandomizedSearchCV(
                estimator,
                param_distributions=param_dist,
                n_iter=n_iter_search,
                scoring='neg_root_mean_squared_error',
                cv=tss,
                random_state=random_seed,
                n_jobs=1,
                verbose=0
            )
            try:
                rs.fit(X_train, y_train)
            except Exception:
                continue
            score = -rs.best_score_
            if best_score is None or score < best_score:
                best_score = score
                best_grid = rs.best_params_.copy()
                best_grid['boosting_type'] = boost
        if best_grid:
            best_params.update(best_grid)
            print(f"[train] tuning selected params: {best_grid} (val_rmse ~ {best_score:.6f})")
        else:
            print("[train] tuning failed or skipped, using default params")

    # enforce gbdt to keep early-stopping behavior stable
    best_params['boosting_type'] = 'gbdt'

    print(f"[train] training LightGBM with params: {best_params}")
    # prepare LightGBM datasets
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_valid = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    lgb_params = {
         "objective": "regression",
         "metric": "rmse",
         "num_leaves": int(best_params.get("num_leaves", 16)),
         "learning_rate": float(best_params.get("learning_rate", 0.03)),
         "feature_fraction": float(best_params.get("feature_fraction", 0.8)),
         "bagging_fraction": float(best_params.get("bagging_fraction", 0.8)),
         "bagging_freq": int(best_params.get("bagging_freq", 5)),
         # regularization
         "min_data_in_leaf": int(best_params.get("min_data_in_leaf", 20)),
         "min_gain_to_split": float(best_params.get("min_gain_to_split", 0.01)),
         "reg_alpha": float(best_params.get("reg_alpha", 0.1)),
         "reg_lambda": float(best_params.get("reg_lambda", 0.1)),
         "verbosity": -1,
         "seed": int(best_params.get("random_state", 42)),
         "boosting_type": best_params.get("boosting_type", "gbdt")
     }

    # determine number of rounds and cap to reasonable maximum (smaller for small data)
    num_round = int(best_params.get("n_estimators", 120))
    num_round = min(num_round, 300)

    # time-limit callback: safely stop training if it runs too long
    import time
    MAX_TRAIN_SECONDS = int(best_params.get("max_train_seconds", 600))  # default 10 minutes
    def _time_limit_callback(max_seconds: int):
        start_time = time.time()
        def callback(env):
            # env.iteration is current iteration (0-based)
            if time.time() - start_time > max_seconds:
                print(f"[train] time limit {max_seconds}s reached, stopping training at iter {env.iteration}")
                try:
                    env.model.stop_training = True
                except Exception:
                    # fallback: raise to break out
                    raise lgb.callback.EarlyStopException(env.iteration)
        callback.order = 30
        return callback

    # use callbacks for early stopping (compatible across lightgbm versions)
    callbacks = [lgb.callback.early_stopping(50), _time_limit_callback(MAX_TRAIN_SECONDS)]
    # include periodic logging via callback (avoid passing verbose_eval kwarg)
    verbose_interval = int(best_params.get("verbose_eval", 50) or 0)
    if verbose_interval > 0:
        try:
            callbacks.append(lgb.callback.log_evaluation(period=verbose_interval))
        except Exception:
            # some lightgbm versions have different callback signatures; ignore if unavailable
            pass

    booster = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=num_round,
        valid_sets=[lgb_train, lgb_valid],
        valid_names=['train', 'valid'],
        callbacks=callbacks
    )

    # validation metrics (use best_iteration if available)
    best_iter = getattr(booster, "best_iteration", None) or num_round
    y_pred_val = booster.predict(X_val, num_iteration=best_iter)
    from sklearn.metrics import mean_absolute_error
    val_rmse = float(np.sqrt(mean_squared_error(y_val, y_pred_val)))
    val_mae = float(mean_absolute_error(y_val, y_pred_val))

    # expose model variable as booster for saving
    model = booster

    # archive existing model and save new booster (timestamped in archived)
    _archive_if_exists(LATEST_MODEL_PATH, ARCHIVED_MODELS, "lightgbm_return_model.pkl")
    LATEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, LATEST_MODEL_PATH)
    print(f"[train] saved model (booster): {LATEST_MODEL_PATH.resolve()}")

    # --- Save canonical feature names with the model and under training metrics ---
    try:
        feature_names = X.columns.tolist()
        # save next to model for quick lookup by inference code
        features_file = LATEST_MODEL_DIR / "feature_names.json"
        with open(features_file, "w") as _f:
            json.dump(feature_names, _f, indent=2)
        print(f"[train] saved feature names: {features_file.resolve()}")

        # also persist into training metrics directory for discoverability
        METRICS_LATEST_DIR.mkdir(parents=True, exist_ok=True)
        metrics_features_file = METRICS_LATEST_DIR / "training_features.json"
        with open(metrics_features_file, "w") as _mf:
            json.dump({"features": feature_names}, _mf, indent=2)
        print(f"[train] saved training features metadata: {metrics_features_file.resolve()}")
    except Exception as _fe_err:
        print(f"[train] failed to save feature names: {_fe_err}")

    # feature importance from booster
    try:
        feat_names = X.columns.tolist()
        fi_gain = np.array(model.feature_importance(importance_type="gain"))
        fi_split = np.array(model.feature_importance(importance_type="split"))

        # Prefer gain when informative, otherwise use split.
        if fi_gain.sum() > 0:
            scores = fi_gain.astype(float)
            source = "gain"
        elif fi_split.sum() > 0:
            scores = fi_split.astype(float)
            source = "split"
        else:
            # no informative importance from model; keep zeros (do NOT assign equal weights)
            scores = np.zeros(len(feat_names), dtype=float)
            source = "none"

        # Normalize to percentages when there is some signal; otherwise keep zeros
        if scores.sum() > 0:
            norm = (scores / scores.sum()) * 100.0
            norm = norm.round(6)
            feat_imp = dict(zip(feat_names, norm.tolist()))
        else:
            feat_imp = dict(zip(feat_names, [0.0] * len(feat_names)))

        print(f"[train] feature importance computed using '{source}'")

        # If there are zero-importance features and at least one positive-importance feature,
        # drop exact-zero features and retrain once on the reduced set.
        zero_feats = [c for c, v in feat_imp.items() if v == 0.0]
        pos_feats = [c for c, v in feat_imp.items() if v > 0.0]
        if zero_feats and pos_feats and (len(pos_feats) >= 3):
            print(f"[train] dropping {len(zero_feats)} zero-importance features: {zero_feats[:10]}{'...' if len(zero_feats)>10 else ''}")
            keep_cols = pos_feats
            X_train_re = X_train[keep_cols]
            X_val_re = X_val[keep_cols]
            lgb_train_re = lgb.Dataset(X_train_re, label=y_train)
            lgb_valid_re = lgb.Dataset(X_val_re, label=y_val, reference=lgb_train_re)
            num_round_re = min(150, num_round)
            callbacks_re = [lgb.callback.early_stopping(50), _time_limit_callback(MAX_TRAIN_SECONDS)]
            try:
                booster = lgb.train(
                    lgb_params,
                    lgb_train_re,
                    num_boost_round=num_round_re,
                    valid_sets=[lgb_train_re, lgb_valid_re],
                    valid_names=['train', 'valid'],
                    callbacks=callbacks_re
                )
                model = booster
                # recompute importances using the same preferred type
                fi_gain = np.array(model.feature_importance(importance_type="gain"))
                fi_split = np.array(model.feature_importance(importance_type="split"))
                if fi_gain.sum() > 0:
                    scores = fi_gain.astype(float)
                elif fi_split.sum() > 0:
                    scores = fi_split.astype(float)
                else:
                    scores = np.zeros(len(keep_cols), dtype=float)
                if scores.sum() > 0:
                    norm = (scores / scores.sum()) * 100.0
                    norm = norm.round(6)
                    feat_imp = dict(zip(keep_cols, norm.tolist()))
                else:
                    feat_imp = dict(zip(keep_cols, [0.0] * len(keep_cols)))
                X = X[keep_cols]
                # save updated model after retrain
                joblib.dump(model, LATEST_MODEL_PATH)
                print(f"[train] saved retrained model (reduced features): {LATEST_MODEL_PATH.resolve()}")
            except Exception as _e:
                print(f"[train] retrain on reduced features failed: {_e}")
    except Exception as _e:
        print(f"[train] feature importance extraction failed: {_e}")
        feat_imp = {}

    # archive metrics and save metadata
    _archive_if_exists(METRICS_LATEST_DIR / "metrics.json", METRICS_ARCHIVED_DIR, "metrics.json")
    METRICS_LATEST_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {
         "trained_at": datetime.now().isoformat(),
         "n_train": len(X_train),
         "n_val": len(X_val),
         "features": X.columns.tolist(),
         "params": best_params,
         "val_rmse": val_rmse,
         "val_mae": val_mae,
         "feature_importance": feat_imp
     }

    metrics_path = METRICS_LATEST_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[train] saved training metadata: {metrics_path.resolve()}")

    return {"model_path": str(LATEST_MODEL_PATH), "metrics_path": str(metrics_path), "val_rmse": val_rmse}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()
    train(random_seed=args.seed, test_size_ratio=args.test_size)
