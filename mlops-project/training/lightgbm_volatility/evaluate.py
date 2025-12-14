from pathlib import Path
import json, shutil
from datetime import datetime
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from features import prepare_features

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
LATEST_MODEL_PATH = MODELS_DIR / "latest" / "lightgbm_volatility_model.pkl"

METRICS_DIR = Path(__file__).resolve().parent / "metrics"
METRICS_LATEST = METRICS_DIR / "latest" / "metrics.json"
METRICS_ARCHIVED = METRICS_DIR / "archived"
PLOTS_DIR = METRICS_DIR / "latest" / "plots"

def _archive_if_exists(path: Path, archived_dir: Path, prefix: str):
	if path.exists():
		archived_dir.mkdir(parents=True, exist_ok=True)
		ts = datetime.now().strftime("%Y%m%dT%H%M%S")
		dest = archived_dir / f"{ts}_{prefix}"
		shutil.move(str(path), str(dest))

def evaluate(target_horizon: int = 3):
	X, y, dates, scaler = prepare_features(scale=True, target_horizon=target_horizon)
	split_idx = int(len(X) * 0.8)
	X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
	y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

	if not LATEST_MODEL_PATH.exists():
		raise FileNotFoundError(f"Model not found at {LATEST_MODEL_PATH}")
	model = joblib.load(LATEST_MODEL_PATH)
	# align features
	try:
		feat_names = model.feature_name()
		X_val_used = X_val[feat_names]
	except Exception:
		X_val_used = X_val

	pred = model.predict(X_val_used)
	rmse = float(np.sqrt(mean_squared_error(y_val, pred)))
	mae = float(mean_absolute_error(y_val, pred))
	r2 = float(r2_score(y_val, pred))

	# feature importance plot
	try:
		fi = np.array(model.feature_importance(importance_type='gain'))
		feat_names = X.columns.tolist()
		if fi.sum() > 0:
			idx = np.argsort(fi)[::-1]
			top = min(20, len(feat_names))
			fig, ax = plt.subplots(figsize=(8,6))
			ax.barh([feat_names[i] for i in idx[:top]][::-1], fi[idx[:top]][::-1])
			ax.set_title("Feature importance (gain)")
			plt.tight_layout()
			PLOTS_DIR.mkdir(parents=True, exist_ok=True)
			fp = PLOTS_DIR / "feature_importance.png"
			fig.savefig(fp, dpi=150)
			plt.close(fig)
	except Exception:
		fp = None

	# save metrics
	_archive_if_exists(METRICS_LATEST, METRICS_ARCHIVED, "metrics.json")
	METRICS_LATEST.parent.mkdir(parents=True, exist_ok=True)
	metrics = {
		"evaluated_at": datetime.now().isoformat(),
		"horizon": target_horizon,
		"n_eval": len(y_val),
		"rmse": rmse, "mae": mae, "r2": r2
	}
	with open(METRICS_LATEST, "w") as f:
		json.dump(metrics, f, indent=2)
	print(f"[eval] saved metrics: {METRICS_LATEST.resolve()}")
	print(f"[eval] RMSE: {rmse:.6f}, MAE: {mae:.6f}, R2: {r2:.6f}")
	return metrics

if __name__ == "__main__":
	metrics = evaluate()
	print("Evaluation metrics:", metrics)
