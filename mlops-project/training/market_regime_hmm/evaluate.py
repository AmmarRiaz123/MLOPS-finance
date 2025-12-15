from pathlib import Path
import json
import logging
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features import prepare_features

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
LATEST_MODEL_PATH = MODELS_DIR / "latest" / "market_regime_hmm.pkl"

METRICS_DIR = Path(__file__).resolve().parent / "metrics"
METRICS_LATEST = METRICS_DIR / "latest" / "metrics.json"
METRICS_ARCHIVED = METRICS_DIR / "archived"
PLOTS_DIR = METRICS_DIR / "latest" / "plots"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def _archive_if_exists(path: Path, archived_dir: Path, prefix: str):
	if path.exists():
		archived_dir.mkdir(parents=True, exist_ok=True)
		ts = datetime.now().strftime("%Y%m%dT%H%M%S")
		dest = archived_dir / f"{ts}_{prefix}"
		shutil.move(str(path), str(dest))

def evaluate():
	log = logging.getLogger("market_regime_hmm.eval")
	log.info("Preparing features")
	X, returns, vol_series, dates, scaler = prepare_features()

	if not LATEST_MODEL_PATH.exists():
		raise FileNotFoundError(f"Model not found: {LATEST_MODEL_PATH}")
	obj = joblib.load(LATEST_MODEL_PATH)
	model = obj.get("model", obj) if isinstance(obj, dict) else obj

	# predict regimes
	regimes = model.predict(X.values)
	dfm = pd.DataFrame({"date": dates.values, "regime": regimes, "return": returns.values, "vol_short": vol_series.values}, index=dates.index)

	regime_counts = dfm['regime'].value_counts().sort_index().to_dict()
	avg_return = dfm.groupby('regime')['return'].mean().to_dict()
	avg_vol = dfm.groupby('regime')['vol_short'].mean().to_dict()
	trans_mat = model.transmat_.tolist()

	metrics = {
		"evaluated_at": pd.Timestamp.now().isoformat(),
		"regime_counts": regime_counts,
		"avg_return_per_regime": {int(k): float(v) for k,v in avg_return.items()},
		"avg_vol_per_regime": {int(k): float(v) for k,v in avg_vol.items()},
		"transition_matrix": trans_mat
	}

	# archive & save metrics
	_archive_if_exists(METRICS_LATEST, METRICS_ARCHIVED, "metrics.json")
	METRICS_LATEST.parent.mkdir(parents=True, exist_ok=True)
	with open(METRICS_LATEST, "w") as f:
		json.dump(metrics, f, indent=2)
	log.info(f"Saved metrics: {METRICS_LATEST.resolve()}")

	# plots
	PLOTS_DIR.mkdir(parents=True, exist_ok=True)
	try:
		price = pd.Series(np.exp(returns.cumsum()), index=dates.index)
		fig, ax = plt.subplots(figsize=(12,4))
		for r in sorted(dfm['regime'].unique()):
			mask = dfm['regime']==r
			ax.plot(dfm.index[mask], price[mask], '.', label=f"regime_{r}")
		ax.legend()
		ax.set_title("Price by regime")
		fig.savefig(PLOTS_DIR / "price_by_regime.png", dpi=150)
		plt.close(fig)
	except Exception as _e:
		log.warning(f"Plot failed: {_e}")

	return metrics

if __name__ == "__main__":
	res = evaluate()
	print("Evaluation metrics:", res)
