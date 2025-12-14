from pathlib import Path
import json, shutil
from datetime import datetime
import joblib
import argparse
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

from features import prepare_features

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
LATEST_MODEL_PATH = MODELS_DIR / "latest" / "lightgbm_volatility_model.pkl"
ARCHIVED_DIR = MODELS_DIR / "archived"

METRICS_DIR = Path(__file__).resolve().parent / "metrics"
METRICS_LATEST = METRICS_DIR / "latest" / "metrics.json"
METRICS_ARCHIVED = METRICS_DIR / "archived"

def _archive_if_exists(path: Path, archived_dir: Path, prefix: str):
	if path.exists():
		archived_dir.mkdir(parents=True, exist_ok=True)
		ts = datetime.now().strftime("%Y%m%dT%H%M%S")
		dest = archived_dir / f"{ts}_{prefix}"
		shutil.move(str(path), str(dest))

def train(random_seed: int = 42, test_size_ratio: float = 0.2, tune: bool = True, tune_iter: int = 8):
	print("[train] preparing features...")
	X, y, dates, scaler = prepare_features(scale=True, target_horizon=3)

	# chronological split 80/20
	split_idx = int(len(X) * (1 - test_size_ratio))
	X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
	y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

	# default conservative params
	default = {
		'num_leaves': 16, 'max_depth': 6, 'learning_rate': 0.03,
		'n_estimators': 120, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
		'reg_alpha': 0.1, 'reg_lambda': 0.1, 'min_data_in_leaf': 20,
		'boosting_type': 'gbdt', 'random_state': random_seed
	}
	params = default.copy()

	# small randomized tuning
	if tune:
		from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
		from lightgbm import LGBMRegressor
		tss = TimeSeriesSplit(n_splits=3)
		param_dist = {
			'num_leaves': [12,16,20],
			'max_depth': [3,5,6],
			'learning_rate': [0.01,0.03],
			'n_estimators': [80,100,150],
			'feature_fraction': [0.7,0.8],
			'bagging_fraction': [0.7,0.8],
			'reg_alpha': [0.0,0.1,0.5], 'reg_lambda': [0.0,0.1,0.5]
		}
		est = LGBMRegressor(boosting_type='gbdt', random_state=random_seed, n_jobs=-1)
		rs = RandomizedSearchCV(est, param_dist, n_iter=min(tune_iter,8), scoring='neg_root_mean_squared_error', cv=tss, random_state=random_seed, n_jobs=1, verbose=0)
		try:
			rs.fit(X_train, y_train)
			best = rs.best_params_
			best['boosting_type'] = 'gbdt'
			params.update(best)
			print(f"[train] tuning selected: {best}")
		except Exception as e:
			print("[train] tuning failed:", e)

	# train final booster with lgb.train (callbacks for early stopping)
	lgb_train = lgb.Dataset(X_train, label=y_train)
	lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

	lgb_params = {
		'objective': 'regression', 'metric': 'rmse',
		'num_leaves': int(params.get('num_leaves')), 'learning_rate': float(params.get('learning_rate')),
		'feature_fraction': float(params.get('feature_fraction')), 'bagging_fraction': float(params.get('bagging_fraction')),
		'min_data_in_leaf': int(params.get('min_data_in_leaf')), 'min_gain_to_split': float(params.get('min_gain_to_split',0.0)),
		'reg_alpha': float(params.get('reg_alpha',0.0)), 'reg_lambda': float(params.get('reg_lambda',0.0)),
		'verbosity': -1, 'seed': int(params.get('random_state',42)), 'boosting_type': 'gbdt'
	}

	num_round = int(params.get('n_estimators', 120))
	callbacks = [lgb.callback.early_stopping(50)]
	booster = lgb.train(lgb_params, lgb_train, num_boost_round=num_round, valid_sets=[lgb_train, lgb_val], valid_names=['train','valid'], callbacks=callbacks)

	# validation metrics
	best_iter = getattr(booster, 'best_iteration', None) or num_round
	pred_val = booster.predict(X_val, num_iteration=best_iter)
	rmse = float(np.sqrt(mean_squared_error(y_val, pred_val)))
	mae = float(mean_absolute_error(y_val, pred_val))

	# archive/save model
	_archive_if_exists(LATEST_MODEL_PATH, ARCHIVED_DIR, "lightgbm_volatility_model.pkl")
	LATEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
	joblib.dump(booster, LATEST_MODEL_PATH)
	print(f"[train] saved model: {LATEST_MODEL_PATH.resolve()}")

	# feature importance (gain preferred)
	try:
		fi_gain = np.array(booster.feature_importance(importance_type='gain'))
		feat_names = X.columns.tolist()
		if fi_gain.sum() > 0:
			norm = (fi_gain / fi_gain.sum()) * 100.0
			feat_imp = dict(zip(feat_names, norm.round(6).tolist()))
		else:
			fi_split = np.array(booster.feature_importance(importance_type='split'))
			if fi_split.sum() > 0:
				norm = (fi_split / fi_split.sum()) * 100.0
				feat_imp = dict(zip(feat_names, norm.round(6).tolist()))
			else:
				feat_imp = dict(zip(feat_names, [0.0]*len(feat_names)))
	except Exception:
		feat_imp = {}

	# archive metrics and save
	_archive_if_exists(METRICS_LATEST, METRICS_ARCHIVED, "metrics.json")
	METRICS_LATEST.parent.mkdir(parents=True, exist_ok=True)
	metrics = {
		"trained_at": datetime.now().isoformat(),
		"n_train": len(X_train), "n_val": len(X_val),
		"params": params, "val_rmse": rmse, "val_mae": mae,
		"feature_importance": feat_imp
	}
	with open(METRICS_LATEST, "w") as f:
		json.dump(metrics, f, indent=2)
	print(f"[train] saved metrics: {METRICS_LATEST.resolve()}")

	return {"model_path": str(LATEST_MODEL_PATH), "metrics_path": str(METRICS_LATEST), "val_rmse": rmse}

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--test_size", type=float, default=0.2)
	parser.add_argument("--tune", action="store_true")
	args = parser.parse_args()
	train(random_seed=args.seed, test_size_ratio=args.test_size, tune=args.tune)
